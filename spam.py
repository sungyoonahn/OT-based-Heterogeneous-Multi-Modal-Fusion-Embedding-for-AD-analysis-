import pandas as pd
# from pathlib import Path

# DATA_DIR = Path("/home/prml/RIMA/datasets/")

# # Iterate over each Excel file and save a CSV next to it
# for xlsx_path in DATA_DIR.glob("*.xlsx"):
#     df = pd.read_excel(xlsx_path)
#     csv_path = xlsx_path.with_suffix(".csv")
#     df.to_csv(csv_path, index=False)
#     print(f"Converted {xlsx_path.name} -> {csv_path.name} ({len(df)} rows)")



import nibabel as nib

# Load the NIfTI image
nifti_image = nib.load('/home/prml/RIMA/datasets/ADNI/MRI-PET/AD_MRI_130_FIN/ADNI/002_S_5018/MT1__N3m/2013-11-18_10_41_00.0/I399905/ADNI_002_S_5018_MR_MT1__N3m_Br_20131203113324147_S206234_I399905.nii')

# Get the data array from the image
data = nifti_image.get_fdata()

# Get the shape of the data array, which represents the dimensions
# The shape will typically be (X, Y, Z) for a 3D image,
# or (X, Y, Z, T) for a 4D image (with time dimension)
dimensions = data.shape

# Extract X, Y, and Z dimensions
x_dim = dimensions[0]
y_dim = dimensions[1]
z_dim = dimensions[2]

print(f"X dimension: {x_dim}")
print(f"Y dimension: {y_dim}")
print(f"Z dimension: {z_dim}")

