import glob
import os 
import SimpleITK as sitk
import numpy as np

from dcmrtstruct2nii.adapters.input.abstractinputadapter import AbstractInputAdapter
from dcmrtstruct2nii.exceptions import InvalidFileFormatException


class DcmInputAdapter(AbstractInputAdapter):
    def ingest(self, input_dir, series_id=None, sort:bool=True):
        '''
            Load DICOMs from input_dir to a single 3D image and make sure axial
            direction is on third axis.
            :param input_dir: Input directory where the dicom files are located
            :param series_id: Optional, the Series Instance UID for the image dicoms
            :return: multidimensional array with pixel data, metadata
        '''
        dicom_reader = sitk.ImageSeriesReader()

        if series_id is None:
            series_id = ''
        dicom_file_names = [f for f in glob.glob(os.path.join(input_dir, "*.dcm")) if not (
            os.path.basename(f).lower().startswith("rd") or \
                os.path.basename(f).lower().startswith("rp") or \
                    os.path.basename(f).lower().startswith("rs"))]
        if sort:
            origins = [sitk.ReadImage(name).GetOrigin()[2] for name in dicom_file_names]
            if not(origins == sorted(origins)):
                dicom_file_names = [dicom_file_names[i] for i in np.argsort(origins)]
           
        if not dicom_file_names:
            raise InvalidFileFormatException('Directory {} is not a dicom'.format(input_dir))

        dicom_reader.SetFileNames(dicom_file_names)

        dicom_image = dicom_reader.Execute()

        return dicom_image
