"""Shared constants used across data prep / augment / training / inference."""

CLASS_NAMES = ["collision", "dirt", "plain particle", "scratch"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

# 11 image stems that exist BYTE-IDENTICALLY in both 训练集/负样本 and 测试集/image.
# Competition rule forbids using test data for training — these MUST be excluded
# from data prep / augmentation. See src/audit.py A6 for the leak detection logic.
LEAKED_STEMS = frozenset({
    "B22009S00_004_1_20220728135451799_000008_0_1",
    "B22009S00_004_1_20221011163400631_000009_2_3",
    "B22009S00_004_1_20230618154219713_000008_2_2",
    "B22009S00_004_1_20231127143151435_000016_3_0_1",
    "B22009S00_004_1_20231220163911526_000004_0_2_1",
    "B22009S00_004_2_20230424101547469_000004_3_2",
    "B22009S00_004_2_20230424102202610_000002_1_1",
    "B22009S00_004_2_20230814150045868_000012_1_2",
    "B22009S00_004_2_20230901160601999_00000a_2_2",
    "B22009S00_004_2_20240201141256231_000004_0_2_1",
    "B22009S00_004_3_20231121133748034_000003_3_1_1",
})
