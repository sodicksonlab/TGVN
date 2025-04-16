import pandas as pd 
import numpy as np
import numpy.fft as ft 
import xml.etree.ElementTree as etree
import torch, os, h5py, hashlib
import torchvision.transforms.functional as F
import torch.nn.functional as TF
import cv2
from pathlib import Path
from fastmri import rss, rss_complex, fft2c, ifft2c, complex_abs
from fastmri.data.mri_data import FastMRIRawDataSample, et_query
from fastmri.data.subsample import MaskFunc, RandomMaskFunc, EquiSpacedMaskFunc
from fastmri.data.transforms import complex_center_crop, center_crop_to_smallest, to_tensor, apply_mask, center_crop
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)


# NamedTuple for a sample in Experiments K1, K2, and K3
class VarNetSampleJoint(NamedTuple):
    pd_kspace: torch.Tensor
    pd_mask: torch.Tensor
    pd_num_low_frequencies: Optional[int]
    pd_target: torch.Tensor
    pdfs_kspace: torch.Tensor
    pdfs_mask: torch.Tensor
    pdfs_num_low_frequencies: Optional[int]
    pdfs_target: torch.Tensor
    pd_fname: str
    pdfs_fname: str
    pd_slice_num: int
    pdfs_slice_num: int
    pd_max_value: float
    pdfs_max_value: float
    pd_crop_size: Tuple[int, int]
    pdfs_crop_size: Tuple[int, int]

# NamedTuple for a sample in Experiments B1
class VarNetSampleM4(NamedTuple):
    flair_kspace: torch.Tensor
    flair_mask: torch.Tensor
    flair_num_low_frequencies: Optional[int]
    flair_target: torch.Tensor
    t2_kspace: torch.Tensor
    flair_fname: str
    t2_fname: str
    flair_slice_num: int
    t2_slice_num: int
    flair_max_value: float


class SliceDatasetJoint(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to matching PDw-PDFSw
    MR image slices from the fastMRI knee dataset (K1, K2, and K3).
    """

    def __init__(
        self,
        csv_path: Union[str, Path, os.PathLike],
        challenge: str,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            csv_path: Path to the dataset.
            challenge: "singlecoil" or "multicoil" depending on which challenge
                to use.
            transform: Optional; A callable object that pre-processes the raw
                data into appropriate form. The transform function should take
                'kspace', 'target', 'attributes', 'filename', and 'slice' as
                inputs. 'target' may be null for test data.
        """
        if challenge not in ("singlecoil", "multicoil"):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        self.transform = transform
        self.recons_key = (
            "reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"
        )

        # read the csv file corresponding to the dataset 
        csv = pd.read_csv(csv_path)
        pd_files = [Path(csv.values[:, 0][ind]) for ind in range(len(csv))]
        pdfs_files = [Path(csv.values[:, 1][ind]) for ind in range(len(csv))]

        self.pd_raw_samples = []
        self.pdfs_raw_samples = []
        for fname in pd_files:
            metadata, num_slices = self._retrieve_metadata(fname)
            new_raw_samples = []
            for slice_ind in range(num_slices):
                raw_sample = FastMRIRawDataSample(fname, slice_ind, metadata)
                new_raw_samples.append(raw_sample)
            self.pd_raw_samples += new_raw_samples
            
        for fname in pdfs_files:
            metadata, num_slices = self._retrieve_metadata(fname)
            new_raw_samples = []
            for slice_ind in range(num_slices):
                raw_sample = FastMRIRawDataSample(fname, slice_ind, metadata)
                new_raw_samples.append(raw_sample)
            self.pdfs_raw_samples += new_raw_samples

    def _retrieve_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            et_root = etree.fromstring(hf["ismrmrd_header"][()])

            enc = ["encoding", "encodedSpace", "matrixSize"]
            enc_size = (
                int(et_query(et_root, enc + ["x"])),
                int(et_query(et_root, enc + ["y"])),
                int(et_query(et_root, enc + ["z"])),
            )
            rec = ["encoding", "reconSpace", "matrixSize"]
            recon_size = (
                int(et_query(et_root, rec + ["x"])),
                int(et_query(et_root, rec + ["y"])),
                int(et_query(et_root, rec + ["z"])),
            )

            lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
            enc_limits_center = int(et_query(et_root, lims + ["center"]))
            enc_limits_max = int(et_query(et_root, lims + ["maximum"])) + 1

            padding_left = enc_size[1] // 2 - enc_limits_center
            padding_right = padding_left + enc_limits_max

            num_slices = hf["kspace"].shape[0]

            metadata = {
                "padding_left": padding_left,
                "padding_right": padding_right,
                "encoding_size": enc_size,
                "recon_size": recon_size,
                **hf.attrs,
            }

        return metadata, num_slices

    def __len__(self):
        return len(self.pd_raw_samples)
        
    def __getitem__(self, i: int):
        pd_fname, pd_dataslice, pd_metadata = self.pd_raw_samples[i]
        pdfs_fname, pdfs_dataslice, pdfs_metadata = self.pdfs_raw_samples[i]
        
        with h5py.File(pdfs_fname, "r") as hf:
            pdfs_kspace = hf["kspace"][pdfs_dataslice]
            pdfs_mask = np.asarray(hf["mask"]) if "mask" in hf else None
            pdfs_target = hf[self.recons_key][pdfs_dataslice] if self.recons_key in hf else None
            
            pdfs_attrs = dict(hf.attrs)
            pdfs_attrs.update(pdfs_metadata)

        with h5py.File(pd_fname, "r") as hf:
            pd_kspace = hf["kspace"][pd_dataslice]
            pd_mask = np.asarray(hf["mask"]) if "mask" in hf else None
            pd_target = hf[self.recons_key][pd_dataslice] if self.recons_key in hf else None

            pd_attrs = dict(hf.attrs)
            pd_attrs.update(pd_metadata)        
            
        if self.transform is None:
            sample = (
                pd_kspace, pd_mask, pd_target, pd_fname.name, pd_dataslice, pd_attrs,
                pdfs_kspace, pdfs_mask, pdfs_target, pdfs_fname.name, pdfs_dataslice, pdfs_attrs
            )
            
        else:
            sample = self.transform(
                pd_kspace, pd_mask, pd_target, pd_fname.name, pd_dataslice, pd_attrs,
                pdfs_kspace, pdfs_mask, pdfs_target, pdfs_fname.name, pdfs_dataslice, pdfs_attrs
            )

        return sample


class SliceDatasetM4(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to matching FLAIR-T2w
    MR image slices from the M4Raw brain dataset (B1).
    """

    def __init__(
        self,
        csv_path: Union[str, Path, os.PathLike],
        challenge: str,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            csv_path: Path to the dataset.
            challenge: "singlecoil" or "multicoil" depending on which challenge
                to use.
            transform: Optional; A callable object that pre-processes the raw
                data into appropriate form. The transform function should take
                'kspace', 'target', 'attributes', 'filename', and 'slice' as
                inputs. 'target' may be null for test data.
        """
        if challenge not in ("singlecoil", "multicoil"):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        self.transform = transform
        self.recons_key = (
            "reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"
        )

        # read the csv file corresponding to the dataset 
        csv = pd.read_csv(csv_path)
        prefix = [Path(csv.values[:, 0][ind]) for ind in range(len(csv))]

        self.flair_raw_samples = []
        self.t2_raw_samples = []
        for fname in prefix:
            fname = Path(str(fname) + '_FLAIR.h5')
            metadata, num_slices = self._retrieve_metadata(fname)
            new_raw_samples = []
            for slice_ind in range(num_slices):
                raw_sample = FastMRIRawDataSample(fname, slice_ind, metadata)
                new_raw_samples.append(raw_sample)
            self.flair_raw_samples += new_raw_samples
            
        for fname in prefix:
            fname = Path(str(fname) + '_T201.h5')
            metadata, num_slices = self._retrieve_metadata(fname)
            new_raw_samples = []
            for slice_ind in range(num_slices):
                raw_sample = FastMRIRawDataSample(fname, slice_ind, metadata)
                new_raw_samples.append(raw_sample)
            self.t2_raw_samples += new_raw_samples

    def _retrieve_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            et_root = etree.fromstring(hf["ismrmrd_header"][()])

            enc = ["encoding", "encodedSpace", "matrixSize"]
            enc_size = (
                int(et_query(et_root, enc + ["x"])),
                int(et_query(et_root, enc + ["y"])),
                int(et_query(et_root, enc + ["z"])),
            )
            rec = ["encoding", "reconSpace", "matrixSize"]
            recon_size = (
                int(et_query(et_root, rec + ["x"])),
                int(et_query(et_root, rec + ["y"])),
                int(et_query(et_root, rec + ["z"])),
            )

            lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
            enc_limits_center = int(et_query(et_root, lims + ["center"]))
            enc_limits_max = int(et_query(et_root, lims + ["maximum"])) + 1

            padding_left = enc_size[1] // 2 - enc_limits_center
            padding_right = padding_left + enc_limits_max

            num_slices = hf["kspace"].shape[0]

            metadata = {
                "padding_left": padding_left,
                "padding_right": padding_right,
                "encoding_size": enc_size,
                "recon_size": recon_size,
                **hf.attrs,
            }

        return metadata, num_slices

    def __len__(self):
        return len(self.flair_raw_samples)

    def __getitem__(self, i: int):
        flair_fname, flair_dataslice, flair_metadata = self.flair_raw_samples[i]
        t2_fname, t2_dataslice, t2_metadata = self.t2_raw_samples[i]
        
        with h5py.File(flair_fname, "r") as hf:
            flair_kspace = hf["kspace"][flair_dataslice]
            flair_mask = np.asarray(hf["mask"]) if "mask" in hf else None
            flair_target = hf[self.recons_key][flair_dataslice] if self.recons_key in hf else None
            
            flair_attrs = dict(hf.attrs)
            flair_attrs.update(flair_metadata)

        with h5py.File(t2_fname, "r") as hf:
            t2_kspace = hf["kspace"][t2_dataslice]

            t2_attrs = dict(hf.attrs)
            t2_attrs.update(t2_metadata)        
            
        if self.transform is None:
            sample = (
                flair_kspace, flair_mask, flair_target, flair_fname.name, flair_dataslice,
                 flair_attrs, t2_kspace, t2_fname.name, t2_dataslice, t2_attrs
            )
            
        else:
            sample = self.transform(
                flair_kspace, flair_mask, flair_target, flair_fname.name, flair_dataslice,
                flair_attrs, t2_kspace, t2_fname.name, t2_dataslice, t2_attrs
            )

        return sample


class VarNetDataTransformM4:
    """
    Data Transformer for training VN-TGVN models for the M4Raw dataset.
    """

    def __init__(
        self,
        flair_mask_func: Optional[MaskFunc] = None,
        use_seed: bool = True
    ):
        """
        Args:
            mask_func: Optional; A function that can create a mask of
                appropriate shape. Defaults to None.
            use_seed: If True, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        self.flair_mask_func = flair_mask_func
        self.use_seed = use_seed

    def __call__(
        self,
        flair_kspace: np.ndarray,
        flair_mask: np.ndarray,
        flair_target: Optional[np.ndarray],
        flair_fname: str,
        flair_slice_num: int, 
        flair_attrs: Dict, 
        t2_kspace: np.ndarray,
        t2_fname: str,
        t2_slice_num: int,
        t2_attrs: Dict,
    ) -> VarNetSampleM4:
        
        if flair_target is not None:
            flair_target_torch = to_tensor(flair_target)
            flair_max_value = flair_attrs["max"]        
        else:
            flair_target_torch = torch.tensor(0)
            flair_max_value = 0.0

        flair_kspace_torch = to_tensor(flair_kspace)
        t2_kspace_torch = to_tensor(t2_kspace)

        seed = None if not self.use_seed else tuple(map(ord, flair_fname))
        flair_acq_start = flair_attrs["padding_left"]
        flair_acq_end = flair_attrs["padding_right"]
        
        flair_masked_kspace, flair_mask_torch, flair_num_low_frequencies = apply_mask(
            flair_kspace_torch, self.flair_mask_func, seed=seed, padding=(flair_acq_start, flair_acq_end)
        )
        
        sample = VarNetSampleM4(
            flair_kspace=flair_masked_kspace,
            flair_mask=flair_mask_torch.to(torch.bool),
            flair_num_low_frequencies=flair_num_low_frequencies,
            flair_target=flair_target_torch,
            t2_kspace=t2_kspace_torch,
            flair_fname=flair_fname,
            t2_fname=t2_fname,
            flair_slice_num=flair_slice_num,
            t2_slice_num=t2_slice_num,
            flair_max_value=flair_max_value,
        )
        return sample

class VarNetDataTransformJoint:
    """
    Data Transformer for training VN-TGVN models for the fastMRI knee dataset.
    """

    def __init__(
        self,
        pd_mask_func: Optional[MaskFunc] = None,
        pdfs_mask_func: Optional[MaskFunc] = None,
        use_seed: bool = True
    ):
        """
        Args:
            mask_func: Optional; A function that can create a mask of
                appropriate shape. Defaults to None.
            use_seed: If True, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        self.pd_mask_func = pd_mask_func
        self.pdfs_mask_func = pdfs_mask_func
        self.use_seed = use_seed

    def __call__(
        self,
        pd_kspace: np.ndarray,
        pd_mask: np.ndarray,
        pd_target: Optional[np.ndarray],
        pd_fname: str,
        pd_slice_num: int, 
        pd_attrs: Dict, 
        pdfs_kspace: np.ndarray,
        pdfs_mask: np.ndarray,
        pdfs_target: Optional[np.ndarray],
        pdfs_fname: str,
        pdfs_slice_num: int,
        pdfs_attrs: Dict,
    ) -> VarNetSampleJoint:
        
        if pd_target is not None and pdfs_target is not None:
            pd_target_torch = to_tensor(pd_target)
            pd_max_value = pd_attrs["max"]

            pdfs_target_torch = to_tensor(pdfs_target)
            pdfs_max_value = pdfs_attrs["max"]           
        else:
            pd_target_torch = torch.tensor(0)
            pdfs_target_torch = torch.tensor(0)
            pd_max_value = 0.0
            pdfs_max_value = 0.0

        pd_kspace_torch = to_tensor(pd_kspace)
        pdfs_kspace_torch = to_tensor(pdfs_kspace)

        seed = None if not self.use_seed else tuple(map(ord, pd_fname))
        pd_acq_start = pd_attrs["padding_left"]
        pd_acq_end = pd_attrs["padding_right"]
        pd_crop_size = (pd_attrs["recon_size"][0], pd_attrs["recon_size"][1])
        
        pdfs_acq_start = pdfs_attrs["padding_left"]
        pdfs_acq_end = pdfs_attrs["padding_right"]
        pdfs_crop_size = (pdfs_attrs["recon_size"][0], pdfs_attrs["recon_size"][1])
        
        pd_masked_kspace, pd_mask_torch, pd_num_low_frequencies = apply_mask(
            pd_kspace_torch, self.pd_mask_func, seed=seed, padding=(pd_acq_start, pd_acq_end)
        )
        
        pdfs_masked_kspace, pdfs_mask_torch, pdfs_num_low_frequencies = apply_mask(
            pdfs_kspace_torch, self.pdfs_mask_func, seed=seed, padding=(pdfs_acq_start, pdfs_acq_end)
        )
        
        sample = VarNetSampleJoint(
            pd_kspace=pd_masked_kspace,
            pd_mask=pd_mask_torch.to(torch.bool),
            pd_num_low_frequencies=pd_num_low_frequencies,
            pd_target=pd_target_torch,
            pdfs_kspace=pdfs_masked_kspace,
            pdfs_mask=pdfs_mask_torch.to(torch.bool),
            pdfs_num_low_frequencies=pdfs_num_low_frequencies,
            pdfs_target=pdfs_target_torch,
            pd_fname=pd_fname,
            pdfs_fname=pdfs_fname,
            pd_slice_num=pd_slice_num,
            pdfs_slice_num=pdfs_slice_num,
            pd_max_value=pd_max_value,
            pdfs_max_value=pdfs_max_value,
            pd_crop_size=pd_crop_size,
            pdfs_crop_size=pdfs_crop_size,
        )
        return sample
