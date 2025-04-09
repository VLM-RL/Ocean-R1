import datasets
# train dataset
ds_name = "minglingfeng/Ocean_R1_visual_data_stage1"
ds = datasets.load_dataset(ds_name)
ds["train"].to_parquet(f"./train_data/{ds_name}.parquet")

ds_name = "minglingfeng/Ocean_R1_visual_data_stage2"
ds = datasets.load_dataset(ds_name)
ds["train"].to_parquet(f"./train_data/{ds_name}.parquet")


# val dataset
ds_name = "minglingfeng/cvbench_test"
ds = datasets.load_dataset(ds_name)
ds["test"].to_parquet(f"./eval_data/{ds_name}.parquet")

ds_name = "minglingfeng/geoqa_test"
ds = datasets.load_dataset(ds_name)
ds["test"].to_parquet(f"./eval_data/{ds_name}.parquet")

# save images
"""
from io import BytesIO
from PIL import Image
image = ds["test"][0]["images"][0]
image = Image.open(BytesIO(image['bytes']))
# save to file
image.save("./geoqa_test_idx_0.png", format='JPEG')
"""
