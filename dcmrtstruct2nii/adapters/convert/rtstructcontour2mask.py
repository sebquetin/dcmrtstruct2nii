import tqdm
import numpy as np
from skimage import draw
import SimpleITK as sitk
import logging
import multiprocessing as mp

from dcmrtstruct2nii.exceptions import ContourOutOfBoundsException


def _process_slice(args):
    """Worker to process all contours for a single Z slice."""
    z, contours, shape, mask_background, mask_foreground = args

    slice_mask = np.full((shape[1], shape[0]), mask_background, dtype=np.uint8)

    for contour, pts in contours:
        filled_poly = draw.polygon2mask(
            (shape[1], shape[0]),
            np.column_stack((pts[:, 1], pts[:, 0]))
        )

        new_mask = np.logical_xor(slice_mask == mask_foreground, filled_poly)
        slice_mask = np.where(new_mask, mask_foreground, mask_background)

    return z, slice_mask


class DcmPatientCoords2Mask():
    def _poly2mask(self, coords_x, coords_y, shape):
        return draw.polygon2mask(tuple(reversed(shape)),
                                 np.column_stack((coords_y, coords_x)))

    def convert(self, rtstruct_contours, dicom_image, mask_background, mask_foreground, multiprocessing:bool=True):
        shape = dicom_image.GetSize()

        mask = sitk.Image(shape, sitk.sitkUInt8)
        mask.CopyInformation(dicom_image)

        np_mask = sitk.GetArrayFromImage(mask)
        np_mask.fill(mask_background)

        slice_dict = {}

        for contour in tqdm.tqdm(rtstruct_contours,
                                 total=len(rtstruct_contours),
                                 desc="Preparing contours"):

            if contour['type'].upper().replace('_', '').strip() not in [
                'CLOSEDPLANAR', 'INTERPOLATEDPLANAR', 'CLOSEDPLANARXOR'
            ]:
                name = contour.get("name", "unnamed")
                logging.info(f'Skipping contour {name}, unsupported type: {contour["type"]}')
                continue

            coordinates = contour['points']
            pts = np.zeros([len(coordinates['x']), 3])

            for index in range(len(coordinates['x'])):
                world_coords = dicom_image.TransformPhysicalPointToContinuousIndex(
                    (coordinates['x'][index],
                     coordinates['y'][index],
                     coordinates['z'][index])
                )
                pts[index] = world_coords

            z = int(round(pts[0, 2]))

            # keep contour + precalculated pts
            slice_dict.setdefault(z, []).append((contour, pts))

        
        args = [
            (z, contours, shape, mask_background, mask_foreground)
            for z, contours in slice_dict.items()
        ]
        if multiprocessing:
            # each slice processed in parallel
            with mp.Pool(mp.cpu_count()) as pool:
                results = list(tqdm.tqdm(
                    pool.imap_unordered(_process_slice, args),
                    total=len(args),
                    desc="Converting contours to mask (parallel)"
                ))
        else:
            results = []
            for arg in tqdm.tqdm(args, total=len(args), desc="Converting contours to mask (sequential)"):
                results.append(_process_slice(arg))
                

        # MERGE RESULTS
        for z, slice_mask in results:
            if 0 <= z < np_mask.shape[0]:
                np_mask[z] = slice_mask
            else:
                raise ContourOutOfBoundsException()

        mask = sitk.GetImageFromArray(np_mask)  # Avoid redundant calls by moving this here
        return mask
