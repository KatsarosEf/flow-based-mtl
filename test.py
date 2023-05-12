import os
import torch
import tqdm
import wandb
from tqdm import tqdm

from argparse import ArgumentParser
from utils.dataset import MTL_TestDataset
from torch.utils.data import DataLoader
from torchvision import transforms

from metrics import SegmentationMetrics, DeblurringMetrics, OpticalFlowMetrics
from models.MIMOUNet.MIMOUNet import VideoMIMOUNet
from utils.transforms import ToTensor, Normalize
import cv2, kornia
import torch.nn.functional as F
from torchvision.utils import flow_to_image


def save_outputs(args, seq_name, name, output_dict):
    #masks
    cv2.imwrite('{}/{}'.format(os.path.join(args.out, seq_name, 'masks'), name),
                torch.argmax(output_dict['segment'][2], 1)[0].cpu().numpy() * 255.0)
    # #images
    cv2.imwrite('{}/{}'.format(os.path.join(args.out, seq_name, 'images'), name[:-4]+'.png'),
                output_dict['deblur'][2][0].permute(1,2,0).cpu().numpy()*255.0)
    ## flows
    cv2.imwrite('{}/{}'.format(os.path.join(args.out, seq_name, 'flows'), name[:-4] + '.png'),
                flow_to_image(output_dict['flow'])[0].permute(1, 2, 0).cpu().numpy())






def evaluate(args, dataloader, model, metrics_dict):

    tasks = model.module.tasks
    metrics = [k for task in tasks for k in metrics_dict[task].metrics]
    metric_cumltive = {k: [] for k in metrics}
    metrics_hl = {k: [] for k in metrics}
    model.train()

    l = ["f793c363-b270-4e47-b8c6-03752a9b56f6_GT_2606.png",
"a0a71e0a-9ad2-4263-b8fc-8381d4efabc0_GT_2687.png",
"25a6b037-4436-4ce0-a9e1-a82a37a32b05_GT_52846.png",
"a43a1946-ab5a-46ab-898e-6bb0b1c32cc9_GT_33715.png",
"ac7da88d-9248-43ed-9f46-4965bfc01c0c_GT_962.png",
"ac7da88d-9248-43ed-9f46-4965bfc01c0c_GT_1411.png",
"a0a71e0a-9ad2-4263-b8fc-8381d4efabc0_GT_2567.png",
"f69085ca-df9a-497f-a80a-236764b99c46_GT_43836.png",
"d69d8dda-722f-4668-962b-0f9fd58a09b2_GT_53.png",
"d0f107bd-867c-4355-8e3b-945ff04446f0_GT_27400.png",
"a1dfdc70-6320-4ab8-8087-cec8adf22546_GT_24759.png",
"ff948169-f061-4d13-8f17-dddd42489350_GT_2020.png",
"fbbba64e-127d-4eb1-a0d3-77af1baa230e_GT_1380.png",
"f69085ca-df9a-497f-a80a-236764b99c46_GT_43291.png",
"f69085ca-df9a-497f-a80a-236764b99c46_GT_42989.png",
"1f3041c2-2971-49eb-95e0-ced6f46e7b6e_GT_22261.png",
"d0f107bd-867c-4355-8e3b-945ff04446f0_GT_25794.png",
"bb9ecc23-47d9-4b69-abf4-fd330eb2a293_GT_1247.png",
"ac7da88d-9248-43ed-9f46-4965bfc01c0c_GT_114.png",
"bcbd1d1e-cdbb-4958-8a88-7e76578d0146_GT_4762.png",
"f8a684b8-4991-48e3-8c74-77371485cac0_GT_478.png",
"8b6d4762-2c7d-49f5-94d0-3770513aa2c6_GT_33484.png",
"d9715e62-092c-4517-ba57-8b7eed6e18a9_GT_2448.png",
"a0a71e0a-9ad2-4263-b8fc-8381d4efabc0_GT_1934.png",
"280b44bd-b97a-4734-955b-d9943642d489_GT_12757.png",
"259402d0-a583-4a45-98e3-b9c95878b436_GT_38295.png",
"4f9143ad-e811-4351-b888-c6445e5c26a4_GT_35025.png",
"ac7da88d-9248-43ed-9f46-4965bfc01c0c_GT_4050.png",
"efccce8a-7bae-46d4-8e1e-8595931756c3_GT_1799.png",
"280b44bd-b97a-4734-955b-d9943642d489_GT_12835.png",
"bc62eb3d-2232-44a0-9db8-a06b7d783843_GT_5083.png",
"f8a684b8-4991-48e3-8c74-77371485cac0_GT_570.png",
"efccce8a-7bae-46d4-8e1e-8595931756c3_GT_2495.png",
"3fc66c96-e402-4e24-9e61-227be00c1d6c_GT_28118.png",
"bcbd1d1e-cdbb-4958-8a88-7e76578d0146_GT_5643.png",
"e93674d6-6420-4434-86a8-cb3837fc45b5_GT_4178.png",
"d32b2069-4c61-4bdd-8857-ccf368695c85_GT_29873.png",
"d9246392-aafd-4fc3-9a98-2e086b40f7d3_GT_5465.png",
"f1eec0c6-b578-4dd3-9972-3fee1d22ad95_GT_10111.png",
"a6d85655-1fdf-420a-ac58-36aed2696bec_GT_457.png",
"a0a71e0a-9ad2-4263-b8fc-8381d4efabc0_GT_1056.png",
"e93674d6-6420-4434-86a8-cb3837fc45b5_GT_6037.png",
"d69d8dda-722f-4668-962b-0f9fd58a09b2_GT_690.png",
"bcbd1d1e-cdbb-4958-8a88-7e76578d0146_GT_5499.png",
"d0b3f9f3-7887-4b0c-91d1-14c67535019f_GT_1364.png",
"d0f107bd-867c-4355-8e3b-945ff04446f0_GT_27445.png",
"34259f85-ded8-4748-8c7c-600383a53c4c_GT_14038.png",
"f821e60c-62d2-485c-8720-ad9c8530ba34_GT_17262.png",
"c7d6c79f-924a-4231-a3bf-303b66d30f2c_GT_230.png",
"d0828337-193e-4f57-8e90-a4a2e2947d9d_GT_20148.png",
"e93674d6-6420-4434-86a8-cb3837fc45b5_GT_5315.png",
"1d1747e8-1d23-4108-98da-4aa158386c54_GT_6879.png",
"f69085ca-df9a-497f-a80a-236764b99c46_GT_43394.png",
"be278a80-e70a-42da-8e0a-b4e36732bf16_GT_8467.png",
"bcbd1d1e-cdbb-4958-8a88-7e76578d0146_GT_5098.png",
"b2677b41-7cea-4181-ab09-43f60bf5ebc1_GT_1763.png",
"efccce8a-7bae-46d4-8e1e-8595931756c3_GT_2328.png",
"bc5a7acf-43c1-40a4-82ea-56318b78f5f6_GT_12373.png",
"3320384c-03dd-49d4-898c-e35f10eeb122_GT_493.png",
"3fc66c96-e402-4e24-9e61-227be00c1d6c_GT_28166.png",
"c7d6c79f-924a-4231-a3bf-303b66d30f2c_GT_1040.png",
"a43a1946-ab5a-46ab-898e-6bb0b1c32cc9_GT_33668.png",
"f921a4fe-bbe2-4f29-90e7-40c024363840_GT_5536.png",
"0c5367f2-b43c-4af3-9751-0219c48796e9_GT_278.png",
"b13bcaf6-f79c-4880-87cd-c7f1e8c72341_GT_6330.png",
"25a6b037-4436-4ce0-a9e1-a82a37a32b05_GT_51999.png",
"1d1747e8-1d23-4108-98da-4aa158386c54_GT_7202.png",
"bcbd1d1e-cdbb-4958-8a88-7e76578d0146_GT_4908.png",
"a0a71e0a-9ad2-4263-b8fc-8381d4efabc0_GT_796.png",
"e54ecd80-4a03-4098-85ff-d8986cd93182_GT_1118.png",
"caa6110f-f2a3-410e-8db8-93bf43665a56_GT_366.png",
"e93674d6-6420-4434-86a8-cb3837fc45b5_GT_5206.png",
"a6d85655-1fdf-420a-ac58-36aed2696bec_GT_276.png",
"a0a71e0a-9ad2-4263-b8fc-8381d4efabc0_GT_1780.png",
"a0a71e0a-9ad2-4263-b8fc-8381d4efabc0_GT_2132.png",
"e93674d6-6420-4434-86a8-cb3837fc45b5_GT_4899.png",
"fb60ca72-0a2b-45de-ad8a-f99cb6dd4910_GT_3534.png",
"3a0d2b4c-52ca-4212-a109-b2309241f783_GT_819.png",
"0c5367f2-b43c-4af3-9751-0219c48796e9_GT_532.png",
"e93674d6-6420-4434-86a8-cb3837fc45b5_GT_5088.png",
"25a6b037-4436-4ce0-a9e1-a82a37a32b05_GT_52717.png",
"efccce8a-7bae-46d4-8e1e-8595931756c3_GT_1709.png",
"bdb78738-396e-43a8-971f-e05635528dde_GT_25715.png",
"ff948169-f061-4d13-8f17-dddd42489350_GT_1267.png",
"3320384c-03dd-49d4-898c-e35f10eeb122_GT_1901.png",
"f69085ca-df9a-497f-a80a-236764b99c46_GT_42761.png",
"fb60ca72-0a2b-45de-ad8a-f99cb6dd4910_GT_3186.png",
"a9ed416e-7bb6-45bb-ad59-4205aee0c7d2_GT_127.png",
"25a6b037-4436-4ce0-a9e1-a82a37a32b05_GT_52090.png",
"34259f85-ded8-4748-8c7c-600383a53c4c_GT_13889.png",
"f69085ca-df9a-497f-a80a-236764b99c46_GT_44474.png",
"b2677b41-7cea-4181-ab09-43f60bf5ebc1_GT_1362.png",
"f69085ca-df9a-497f-a80a-236764b99c46_GT_44297.png",
"e54ecd80-4a03-4098-85ff-d8986cd93182_GT_1056.png",
"bc5a7acf-43c1-40a4-82ea-56318b78f5f6_GT_11930.png",
"f793c363-b270-4e47-b8c6-03752a9b56f6_GT_2216.png",
"12fd0a0c-7c96-4335-b834-6a8b414a2597_GT_3975.png",
"a9a70fdf-50b6-46f0-8b5a-e24cb90d8f74_GT_29888.png",
"b0c666e9-7ca4-4227-ae0b-f80bee479e69_GT_773.png",
"3a0d2b4c-52ca-4212-a109-b2309241f783_GT_1022.png",
"b0c666e9-7ca4-4227-ae0b-f80bee479e69_GT_2117.png",
"d0b3f9f3-7887-4b0c-91d1-14c67535019f_GT_609.png",
"33c32af3-b069-43d3-85a6-939e3db2976a_GT_3021.png",
"b2677b41-7cea-4181-ab09-43f60bf5ebc1_GT_1930.png",
"d37a25db-fa3f-4144-9a61-45d920a2f2e7_GT_97497.png",
"fbbba64e-127d-4eb1-a0d3-77af1baa230e_GT_1778.png",
"be2c246a-1386-4999-b250-dd29ed503756_GT_31.png",
"f69085ca-df9a-497f-a80a-236764b99c46_GT_43228.png",
"f69085ca-df9a-497f-a80a-236764b99c46_GT_44389.png",
"8b6d4762-2c7d-49f5-94d0-3770513aa2c6_GT_32907.png",
"f69085ca-df9a-497f-a80a-236764b99c46_GT_43486.png",
"a6d85655-1fdf-420a-ac58-36aed2696bec_GT_617.png",
"bc5a7acf-43c1-40a4-82ea-56318b78f5f6_GT_12448.png",
"f8a684b8-4991-48e3-8c74-77371485cac0_GT_347.png",
"efccce8a-7bae-46d4-8e1e-8595931756c3_GT_1882.png",
"a0a71e0a-9ad2-4263-b8fc-8381d4efabc0_GT_2269.png",
"f69085ca-df9a-497f-a80a-236764b99c46_GT_42899.png",
"0be77cff-5c73-4c4a-b788-c36c0b7616c1_GT_10291.png",
"d9246392-aafd-4fc3-9a98-2e086b40f7d3_GT_5017.png",
"904bb444-c07a-4b1a-bcc5-083b7d26de4e_GT_9404.png",
"ff948169-f061-4d13-8f17-dddd42489350_GT_1514.png",
"f1aa290e-0de4-4ccf-9e85-378d13b541c1_GT_35550.png",
"25a6b037-4436-4ce0-a9e1-a82a37a32b05_GT_51991.png",
"f921a4fe-bbe2-4f29-90e7-40c024363840_GT_5981.png",
"bdb78738-396e-43a8-971f-e05635528dde_GT_25510.png",
"d0b3f9f3-7887-4b0c-91d1-14c67535019f_GT_887.png",
"f921a4fe-bbe2-4f29-90e7-40c024363840_GT_5686.png",
"bb9ecc23-47d9-4b69-abf4-fd330eb2a293_GT_856.png",
"12fd0a0c-7c96-4335-b834-6a8b414a2597_GT_4218.png",
"b2677b41-7cea-4181-ab09-43f60bf5ebc1_GT_1632.png",
"3320384c-03dd-49d4-898c-e35f10eeb122_GT_1053.png",
"25a6b037-4436-4ce0-a9e1-a82a37a32b05_GT_52505.png",
"e783404f-9169-4011-be6d-6f729282da43_GT_149.png",
"a9ed416e-7bb6-45bb-ad59-4205aee0c7d2_GT_227.png",
"d9246392-aafd-4fc3-9a98-2e086b40f7d3_GT_4902.png",
"fb60ca72-0a2b-45de-ad8a-f99cb6dd4910_GT_3374.png",
"25a6b037-4436-4ce0-a9e1-a82a37a32b05_GT_52317.png",
"f69085ca-df9a-497f-a80a-236764b99c46_GT_43381.png",
"d9715e62-092c-4517-ba57-8b7eed6e18a9_GT_3164.png",
"f6097e3c-821f-4602-abd2-e927adae4f41_GT_14497.png",
"114d4887-8177-4682-ace8-5275ba137757_GT_14153.png",
"ac7da88d-9248-43ed-9f46-4965bfc01c0c_GT_3537.png",
"2684e679-e335-4563-aef8-5f9c5000dd18_GT_347.png",
"e54ecd80-4a03-4098-85ff-d8986cd93182_GT_975.png",
"d9715e62-092c-4517-ba57-8b7eed6e18a9_GT_1535.png",
"e54ecd80-4a03-4098-85ff-d8986cd93182_GT_302.png",
"3a0d2b4c-52ca-4212-a109-b2309241f783_GT_303.png",
"d0828337-193e-4f57-8e90-a4a2e2947d9d_GT_19600.png",
"f793c363-b270-4e47-b8c6-03752a9b56f6_GT_2335.png",
"a6d85655-1fdf-420a-ac58-36aed2696bec_GT_682.png",
"d9715e62-092c-4517-ba57-8b7eed6e18a9_GT_1803.png",
"b0c666e9-7ca4-4227-ae0b-f80bee479e69_GT_2202.png",
"fe78c8b8-765a-44da-a10f-09a0651f3a6b_GT_41072.png",
"2684e679-e335-4563-aef8-5f9c5000dd18_GT_531.png",
"d9715e62-092c-4517-ba57-8b7eed6e18a9_GT_1564.png",
"3a0d2b4c-52ca-4212-a109-b2309241f783_GT_1353.png",
"3fc66c96-e402-4e24-9e61-227be00c1d6c_GT_27489.png",
"12fd0a0c-7c96-4335-b834-6a8b414a2597_GT_4259.png",
"4470f761-7f1f-4cb1-aa8a-0dcc71e97231_GT_212.png",
"bc5a7acf-43c1-40a4-82ea-56318b78f5f6_GT_12078.png",
"bcbd1d1e-cdbb-4958-8a88-7e76578d0146_GT_5995.png",
"fbbba64e-127d-4eb1-a0d3-77af1baa230e_GT_1025.png",
"be278a80-e70a-42da-8e0a-b4e36732bf16_GT_8615.png",
"a9d74f6a-2e98-4a5e-99ed-aac9ba7ee452_GT_53708.png",
"bcbd1d1e-cdbb-4958-8a88-7e76578d0146_GT_4708.png",
"3fc66c96-e402-4e24-9e61-227be00c1d6c_GT_27340.png",
"ac7da88d-9248-43ed-9f46-4965bfc01c0c_GT_879.png",
"e783404f-9169-4011-be6d-6f729282da43_GT_791.png",
"cceffabf-1649-4e65-80cc-be9aa2db8af4_GT_14.png",
"b90cfc2f-7811-4d91-a929-7843781a2edb_GT_22047.png",
"ac7da88d-9248-43ed-9f46-4965bfc01c0c_GT_2731.png",
"bcbd1d1e-cdbb-4958-8a88-7e76578d0146_GT_5872.png",
"fb60ca72-0a2b-45de-ad8a-f99cb6dd4910_GT_3660.png",
"4d4196a1-c50c-4465-92d3-29b9c9fb541b_GT_2340.png",
"0f6febd6-2913-477a-872b-6fe5dc0a95de_GT_12686.png",
"bc5a7acf-43c1-40a4-82ea-56318b78f5f6_GT_12758.png",
"904bb444-c07a-4b1a-bcc5-083b7d26de4e_GT_9460.png",
"1f3041c2-2971-49eb-95e0-ced6f46e7b6e_GT_22347.png",
"ac7da88d-9248-43ed-9f46-4965bfc01c0c_GT_1953.png",
"ac7da88d-9248-43ed-9f46-4965bfc01c0c_GT_1116.png",
"e54ecd80-4a03-4098-85ff-d8986cd93182_GT_611.png",
"8b6d4762-2c7d-49f5-94d0-3770513aa2c6_GT_33285.png",
"efccce8a-7bae-46d4-8e1e-8595931756c3_GT_2220.png",
"f50d66cf-6481-4f71-963c-fdae3e2da7f8_GT_382.png",
"d0f107bd-867c-4355-8e3b-945ff04446f0_GT_26382.png",
"fb60ca72-0a2b-45de-ad8a-f99cb6dd4910_GT_3073.png",
"a847cec6-5bac-4148-a717-7cc5f20d1cd6_GT_96.png",
"01ff3155-8ab4-47f8-b24d-3eee186be0d6_GT_90767.png",
"b0c666e9-7ca4-4227-ae0b-f80bee479e69_GT_137.png",
"ec7ddfa9-5f5f-47e1-aadb-ad0222ae6230_GT_368.png",
"8b6d4762-2c7d-49f5-94d0-3770513aa2c6_GT_33097.png",
"fbbba64e-127d-4eb1-a0d3-77af1baa230e_GT_1305.png",
"bcbd1d1e-cdbb-4958-8a88-7e76578d0146_GT_5219.png",
"d9715e62-092c-4517-ba57-8b7eed6e18a9_GT_2085.png",
"bcbd1d1e-cdbb-4958-8a88-7e76578d0146_GT_4338.png",
"e783404f-9169-4011-be6d-6f729282da43_GT_269.png",
"d69d8dda-722f-4668-962b-0f9fd58a09b2_GT_711.png",
"cceffabf-1649-4e65-80cc-be9aa2db8af4_GT_105.png",
"f821e60c-62d2-485c-8720-ad9c8530ba34_GT_16861.png",
"e93674d6-6420-4434-86a8-cb3837fc45b5_GT_4138.png",
"1ce8f237-6c69-491e-b69f-e4f0b2f38580_GT_506.png",
"d32b2069-4c61-4bdd-8857-ccf368695c85_GT_29714.png",
"bcbd1d1e-cdbb-4958-8a88-7e76578d0146_GT_4140.png",
"990e7b14-daa3-4685-ae55-6c4bd2c942a6_GT_496.png",
"bc5a7acf-43c1-40a4-82ea-56318b78f5f6_GT_11818.png",
"1f3041c2-2971-49eb-95e0-ced6f46e7b6e_GT_22993.png",
"d9246392-aafd-4fc3-9a98-2e086b40f7d3_GT_5360.png",
"b2677b41-7cea-4181-ab09-43f60bf5ebc1_GT_33.png",
"4109ae7c-65f1-4d6e-83fd-c007c4567f06_GT_3008.png",
"12fd0a0c-7c96-4335-b834-6a8b414a2597_GT_3803.png",
"bdb78738-396e-43a8-971f-e05635528dde_GT_25628.png",
"f9ba6653-39d2-41d6-9593-c0296071441c_GT_2402.png",
"ed351457-849e-4e27-9c6e-448718e9c4ac_GT_104895.png",
"a0a71e0a-9ad2-4263-b8fc-8381d4efabc0_GT_1390.png",
"1f3041c2-2971-49eb-95e0-ced6f46e7b6e_GT_22650.png",
"dea694d7-d654-461c-8104-39c2692b53a6_GT_78.png",
"c7d6c79f-924a-4231-a3bf-303b66d30f2c_GT_1132.png",
"ac7da88d-9248-43ed-9f46-4965bfc01c0c_GT_3504.png",
"e93674d6-6420-4434-86a8-cb3837fc45b5_GT_5636.png",
"a0a71e0a-9ad2-4263-b8fc-8381d4efabc0_GT_2789.png",
"caa6110f-f2a3-410e-8db8-93bf43665a56_GT_118.png",
"bcbd1d1e-cdbb-4958-8a88-7e76578d0146_GT_4845.png",
"fbbba64e-127d-4eb1-a0d3-77af1baa230e_GT_738.png",
"b8045794-6524-42a6-a01a-07924bf3f32a_GT_4091.png",
"d9715e62-092c-4517-ba57-8b7eed6e18a9_GT_1249.png",
"12fd0a0c-7c96-4335-b834-6a8b414a2597_GT_4032.png",
"990e7b14-daa3-4685-ae55-6c4bd2c942a6_GT_743.png",
"a43a1946-ab5a-46ab-898e-6bb0b1c32cc9_GT_33656.png",
"3fc66c96-e402-4e24-9e61-227be00c1d6c_GT_27811.png",
"25a6b037-4436-4ce0-a9e1-a82a37a32b05_GT_51809.png",
"ac7da88d-9248-43ed-9f46-4965bfc01c0c_GT_253.png",
"f69085ca-df9a-497f-a80a-236764b99c46_GT_44124.png",
"1f3041c2-2971-49eb-95e0-ced6f46e7b6e_GT_23162.png",
"a0a71e0a-9ad2-4263-b8fc-8381d4efabc0_GT_2617.png",
"ac7da88d-9248-43ed-9f46-4965bfc01c0c_GT_3023.png",
"d9715e62-092c-4517-ba57-8b7eed6e18a9_GT_3337.png",
"b0c666e9-7ca4-4227-ae0b-f80bee479e69_GT_188.png",
"d0b3f9f3-7887-4b0c-91d1-14c67535019f_GT_1145.png",
"0c5367f2-b43c-4af3-9751-0219c48796e9_GT_1.png",
"ea3124fb-18b6-4cd4-b610-e8c844f6820b_GT_137.png",
"d9715e62-092c-4517-ba57-8b7eed6e18a9_GT_3029.png",
"c7d6c79f-924a-4231-a3bf-303b66d30f2c_GT_1456.png",
"bcbd1d1e-cdbb-4958-8a88-7e76578d0146_GT_5334.png",
"d9715e62-092c-4517-ba57-8b7eed6e18a9_GT_3539.png",
"bcbd1d1e-cdbb-4958-8a88-7e76578d0146_GT_4386.png",
"bbacdcd5-1a9a-4e2e-ae36-785958c11af0_GT_31236.png",
"d9715e62-092c-4517-ba57-8b7eed6e18a9_GT_1561.png",
"1c3ea3e4-9a71-4c72-b5d0-d6f40f9eece6_GT_65.png",
"b2677b41-7cea-4181-ab09-43f60bf5ebc1_GT_2111.png",
"114d4887-8177-4682-ace8-5275ba137757_GT_14431.png",
"ac7da88d-9248-43ed-9f46-4965bfc01c0c_GT_1701.png",
"d9715e62-092c-4517-ba57-8b7eed6e18a9_GT_3126.png",
"f793c363-b270-4e47-b8c6-03752a9b56f6_GT_1877.png",
"e93674d6-6420-4434-86a8-cb3837fc45b5_GT_5776.png",
"e783404f-9169-4011-be6d-6f729282da43_GT_1088.png",
"ac7da88d-9248-43ed-9f46-4965bfc01c0c_GT_4120.png",
"d823a7a2-cea9-45b9-9c3c-d5309b16fdf5_GT_169.png",
"dea694d7-d654-461c-8104-39c2692b53a6_GT_287.png",
"a43a1946-ab5a-46ab-898e-6bb0b1c32cc9_GT_33202.png",
"ac7da88d-9248-43ed-9f46-4965bfc01c0c_GT_818.png",
"01ff3155-8ab4-47f8-b24d-3eee186be0d6_GT_90587.png",
"b2677b41-7cea-4181-ab09-43f60bf5ebc1_GT_447.png",
"a0a71e0a-9ad2-4263-b8fc-8381d4efabc0_GT_860.png",
"3320384c-03dd-49d4-898c-e35f10eeb122_GT_1954.png",
"d0f107bd-867c-4355-8e3b-945ff04446f0_GT_25657.png",
"ed351457-849e-4e27-9c6e-448718e9c4ac_GT_104694.png",
"25a6b037-4436-4ce0-a9e1-a82a37a32b05_GT_52566.png",
"f1eec0c6-b578-4dd3-9972-3fee1d22ad95_GT_9959.png",
"dea694d7-d654-461c-8104-39c2692b53a6_GT_237.png",
"b0988c4a-3373-472c-8990-db2f6729d100_GT_21618.png",
"259402d0-a583-4a45-98e3-b9c95878b436_GT_38102.png",
"3224744e-98a3-4b6d-8ddf-c9c457f369c6_GT_3983.png",
"ac7da88d-9248-43ed-9f46-4965bfc01c0c_GT_1537.png",
"2684e679-e335-4563-aef8-5f9c5000dd18_GT_873.png",
"ff948169-f061-4d13-8f17-dddd42489350_GT_1377.png",
"ff0478b5-5f2c-4450-8b90-d7450325e351_GT_538.png",
"d32b2069-4c61-4bdd-8857-ccf368695c85_GT_29660.png",
"d9715e62-092c-4517-ba57-8b7eed6e18a9_GT_1364.png",
"f821e60c-62d2-485c-8720-ad9c8530ba34_GT_16781.png",
"3320384c-03dd-49d4-898c-e35f10eeb122_GT_637.png",
"f793c363-b270-4e47-b8c6-03752a9b56f6_GT_2308.png",
"1f3041c2-2971-49eb-95e0-ced6f46e7b6e_GT_22146.png",
"f793c363-b270-4e47-b8c6-03752a9b56f6_GT_2119.png",
"a9ed416e-7bb6-45bb-ad59-4205aee0c7d2_GT_511.png",
"ac7da88d-9248-43ed-9f46-4965bfc01c0c_GT_1180.png",
"1ce8f237-6c69-491e-b69f-e4f0b2f38580_GT_416.png",
"f821e60c-62d2-485c-8720-ad9c8530ba34_GT_17132.png",
"ac7da88d-9248-43ed-9f46-4965bfc01c0c_GT_356.png",
"fb60ca72-0a2b-45de-ad8a-f99cb6dd4910_GT_3714.png",
"114d4887-8177-4682-ace8-5275ba137757_GT_14562.png",
"bc62eb3d-2232-44a0-9db8-a06b7d783843_GT_5011.png",
"25a6b037-4436-4ce0-a9e1-a82a37a32b05_GT_51782.png",
"efccce8a-7bae-46d4-8e1e-8595931756c3_GT_2037.png",
"e54ecd80-4a03-4098-85ff-d8986cd93182_GT_846.png",
"ea3124fb-18b6-4cd4-b610-e8c844f6820b_GT_464.png",
"b0c666e9-7ca4-4227-ae0b-f80bee479e69_GT_1094.png",
"ec7ddfa9-5f5f-47e1-aadb-ad0222ae6230_GT_293.png",
"c7d6c79f-924a-4231-a3bf-303b66d30f2c_GT_739.png",
"1f3041c2-2971-49eb-95e0-ced6f46e7b6e_GT_22058.png",
"f793c363-b270-4e47-b8c6-03752a9b56f6_GT_1954.png",
"d0b3f9f3-7887-4b0c-91d1-14c67535019f_GT_555.png",
"fb60ca72-0a2b-45de-ad8a-f99cb6dd4910_GT_3293.png",
"d9246392-aafd-4fc3-9a98-2e086b40f7d3_GT_5204.png",
"d0f107bd-867c-4355-8e3b-945ff04446f0_GT_27133.png",
"ac7da88d-9248-43ed-9f46-4965bfc01c0c_GT_773.png",
"f6097e3c-821f-4602-abd2-e927adae4f41_GT_14760.png",
"ac7da88d-9248-43ed-9f46-4965bfc01c0c_GT_4358.png",
"b90cfc2f-7811-4d91-a929-7843781a2edb_GT_21403.png",
"b0c666e9-7ca4-4227-ae0b-f80bee479e69_GT_511.png",
"fbbba64e-127d-4eb1-a0d3-77af1baa230e_GT_1703.png",
"ac7da88d-9248-43ed-9f46-4965bfc01c0c_GT_164.png",
"3fc66c96-e402-4e24-9e61-227be00c1d6c_GT_27411.png",
"33c32af3-b069-43d3-85a6-939e3db2976a_GT_2942.png",
"ac7da88d-9248-43ed-9f46-4965bfc01c0c_GT_3108.png",
"ac7da88d-9248-43ed-9f46-4965bfc01c0c_GT_3307.png",
"114d4887-8177-4682-ace8-5275ba137757_GT_14674.png",
"efccce8a-7bae-46d4-8e1e-8595931756c3_GT_2813.png",
"f793c363-b270-4e47-b8c6-03752a9b56f6_GT_2489.png",
"b0c666e9-7ca4-4227-ae0b-f80bee479e69_GT_1577.png",
"bcbd1d1e-cdbb-4958-8a88-7e76578d0146_GT_6168.png",
"fbbba64e-127d-4eb1-a0d3-77af1baa230e_GT_1472.png",
"1f3041c2-2971-49eb-95e0-ced6f46e7b6e_GT_23029.png",
"1f5a5a29-a951-4c68-ba31-b6aefeb96c18_GT_1143.png",
"f8a684b8-4991-48e3-8c74-77371485cac0_GT_613.png",
"f793c363-b270-4e47-b8c6-03752a9b56f6_GT_2358.png",
"ac7da88d-9248-43ed-9f46-4965bfc01c0c_GT_1300.png",
"a43a1946-ab5a-46ab-898e-6bb0b1c32cc9_GT_34371.png",
"bb9ecc23-47d9-4b69-abf4-fd330eb2a293_GT_1386.png",
"0f6febd6-2913-477a-872b-6fe5dc0a95de_GT_12976.png",
"ea3124fb-18b6-4cd4-b610-e8c844f6820b_GT_95.png",
"ea3124fb-18b6-4cd4-b610-e8c844f6820b_GT_414.png",
"f8a684b8-4991-48e3-8c74-77371485cac0_GT_56.png",
"3320384c-03dd-49d4-898c-e35f10eeb122_GT_774.png",
"d69d8dda-722f-4668-962b-0f9fd58a09b2_GT_501.png",
"ec7ddfa9-5f5f-47e1-aadb-ad0222ae6230_GT_432.png",
"e93674d6-6420-4434-86a8-cb3837fc45b5_GT_4458.png",
"caa6110f-f2a3-410e-8db8-93bf43665a56_GT_537.png",
"2684e679-e335-4563-aef8-5f9c5000dd18_GT_180.png",
"c7d6c79f-924a-4231-a3bf-303b66d30f2c_GT_1306.png",
"e54ecd80-4a03-4098-85ff-d8986cd93182_GT_115.png",
"fbbba64e-127d-4eb1-a0d3-77af1baa230e_GT_872.png",
"1f5a5a29-a951-4c68-ba31-b6aefeb96c18_GT_1321.png",
"1f3041c2-2971-49eb-95e0-ced6f46e7b6e_GT_22462.png",
"ba231db0-fd0d-41f9-9c9e-8744f6652e94_GT_3727.png",
"b90cfc2f-7811-4d91-a929-7843781a2edb_GT_22372.png",
"d521befa-8b6c-4c7d-8ce7-a0fc0dea6d70_GT_2833.png",
"caa6110f-f2a3-410e-8db8-93bf43665a56_GT_328.png",
"b90cfc2f-7811-4d91-a929-7843781a2edb_GT_21995.png",
"280b44bd-b97a-4734-955b-d9943642d489_GT_13039.png",
"d0f107bd-867c-4355-8e3b-945ff04446f0_GT_26305.png",
"ac7da88d-9248-43ed-9f46-4965bfc01c0c_GT_1546.png",
"e783404f-9169-4011-be6d-6f729282da43_GT_1051.png",
"b0988c4a-3373-472c-8990-db2f6729d100_GT_21369.png",
"d0f107bd-867c-4355-8e3b-945ff04446f0_GT_27392.png",
"218907bd-d0c8-488b-85d2-933643ec6f4e_GT_10407.png",
"ac7da88d-9248-43ed-9f46-4965bfc01c0c_GT_2408.png",
"12fd0a0c-7c96-4335-b834-6a8b414a2597_GT_4406.png",
"ac7da88d-9248-43ed-9f46-4965bfc01c0c_GT_2518.png",
"efccce8a-7bae-46d4-8e1e-8595931756c3_GT_2858.png",
"d0828337-193e-4f57-8e90-a4a2e2947d9d_GT_19961.png",
"d9715e62-092c-4517-ba57-8b7eed6e18a9_GT_2283.png",
"3320384c-03dd-49d4-898c-e35f10eeb122_GT_1659.png",
"b90cfc2f-7811-4d91-a929-7843781a2edb_GT_22135.png",
"d0f107bd-867c-4355-8e3b-945ff04446f0_GT_26185.png",
"114d4887-8177-4682-ace8-5275ba137757_GT_14852.png",
"f6097e3c-821f-4602-abd2-e927adae4f41_GT_14892.png",
"f793c363-b270-4e47-b8c6-03752a9b56f6_GT_1656.png",
"1ce8f237-6c69-491e-b69f-e4f0b2f38580_GT_303.png",
"f1eec0c6-b578-4dd3-9972-3fee1d22ad95_GT_9707.png",
"ac7da88d-9248-43ed-9f46-4965bfc01c0c_GT_4012.png",
"fbbba64e-127d-4eb1-a0d3-77af1baa230e_GT_2276.png",
"1f3041c2-2971-49eb-95e0-ced6f46e7b6e_GT_22820.png",
"ac7da88d-9248-43ed-9f46-4965bfc01c0c_GT_2766.png",
"e93674d6-6420-4434-86a8-cb3837fc45b5_GT_4580.png",
"990e7b14-daa3-4685-ae55-6c4bd2c942a6_GT_998.png",
"bcbd1d1e-cdbb-4958-8a88-7e76578d0146_GT_6083.png",
"f8a684b8-4991-48e3-8c74-77371485cac0_GT_766.png",
"e783404f-9169-4011-be6d-6f729282da43_GT_1127.png",
"b0c666e9-7ca4-4227-ae0b-f80bee479e69_GT_897.png",
"280b44bd-b97a-4734-955b-d9943642d489_GT_13264.png",
"e783404f-9169-4011-be6d-6f729282da43_GT_898.png",
"d0b3f9f3-7887-4b0c-91d1-14c67535019f_GT_739.png",
"e93674d6-6420-4434-86a8-cb3837fc45b5_GT_6050.png",
"3fc66c96-e402-4e24-9e61-227be00c1d6c_GT_27969.png",
"b2677b41-7cea-4181-ab09-43f60bf5ebc1_GT_262.png",
"0be77cff-5c73-4c4a-b788-c36c0b7616c1_GT_9473.png",
"259402d0-a583-4a45-98e3-b9c95878b436_GT_38381.png",
"f921a4fe-bbe2-4f29-90e7-40c024363840_GT_5621.png",
"a6d85655-1fdf-420a-ac58-36aed2696bec_GT_102.png",
"ff948169-f061-4d13-8f17-dddd42489350_GT_1832.png",
"1c3ea3e4-9a71-4c72-b5d0-d6f40f9eece6_GT_278.png",
"caa6110f-f2a3-410e-8db8-93bf43665a56_GT_742.png",
"a847cec6-5bac-4148-a717-7cc5f20d1cd6_GT_47.png",
"d9246392-aafd-4fc3-9a98-2e086b40f7d3_GT_4802.png",
"f69085ca-df9a-497f-a80a-236764b99c46_GT_44037.png",
"d0f107bd-867c-4355-8e3b-945ff04446f0_GT_26337.png",
"2684e679-e335-4563-aef8-5f9c5000dd18_GT_679.png",
"cceffabf-1649-4e65-80cc-be9aa2db8af4_GT_330.png",
"b90cfc2f-7811-4d91-a929-7843781a2edb_GT_22577.png",
"b0c666e9-7ca4-4227-ae0b-f80bee479e69_GT_1564.png",
"a0a71e0a-9ad2-4263-b8fc-8381d4efabc0_GT_2535.png",
"fbbba64e-127d-4eb1-a0d3-77af1baa230e_GT_753.png",
"3a0d2b4c-52ca-4212-a109-b2309241f783_GT_437.png",
"25a6b037-4436-4ce0-a9e1-a82a37a32b05_GT_52383.png",
"0be77cff-5c73-4c4a-b788-c36c0b7616c1_GT_10144.png",
"4d4196a1-c50c-4465-92d3-29b9c9fb541b_GT_2214.png",
"b90cfc2f-7811-4d91-a929-7843781a2edb_GT_21365.png",
"fbbba64e-127d-4eb1-a0d3-77af1baa230e_GT_1122.png",
"be278a80-e70a-42da-8e0a-b4e36732bf16_GT_8451.png",
"d69d8dda-722f-4668-962b-0f9fd58a09b2_GT_976.png",
"b2677b41-7cea-4181-ab09-43f60bf5ebc1_GT_1217.png",
"c7d6c79f-924a-4231-a3bf-303b66d30f2c_GT_390.png",
"b13bcaf6-f79c-4880-87cd-c7f1e8c72341_GT_6525.png",
"1f5a5a29-a951-4c68-ba31-b6aefeb96c18_GT_897.png",
"a6d85655-1fdf-420a-ac58-36aed2696bec_GT_505.png",
"ac7da88d-9248-43ed-9f46-4965bfc01c0c_GT_2299.png",
"990e7b14-daa3-4685-ae55-6c4bd2c942a6_GT_1065.png",
"a9ed416e-7bb6-45bb-ad59-4205aee0c7d2_GT_644.png",
"b0c666e9-7ca4-4227-ae0b-f80bee479e69_GT_307.png",
"efccce8a-7bae-46d4-8e1e-8595931756c3_GT_2184.png",
"bb9ecc23-47d9-4b69-abf4-fd330eb2a293_GT_1474.png",
"ac7da88d-9248-43ed-9f46-4965bfc01c0c_GT_563.png",
"b092fa3f-cbf3-4939-aa1f-d8130a506a5e_GT_5358.png",
"b0c666e9-7ca4-4227-ae0b-f80bee479e69_GT_1855.png",
"d37a25db-fa3f-4144-9a61-45d920a2f2e7_GT_97581.png",
"bcbd1d1e-cdbb-4958-8a88-7e76578d0146_GT_5689.png",
"ac7da88d-9248-43ed-9f46-4965bfc01c0c_GT_2361.png",
"bc5a7acf-43c1-40a4-82ea-56318b78f5f6_GT_12883.png",
"a847cec6-5bac-4148-a717-7cc5f20d1cd6_GT_16.png",
"ac7da88d-9248-43ed-9f46-4965bfc01c0c_GT_3224.png",
"8b6d4762-2c7d-49f5-94d0-3770513aa2c6_GT_32969.png",
"e93674d6-6420-4434-86a8-cb3837fc45b5_GT_4785.png",
"c7d6c79f-924a-4231-a3bf-303b66d30f2c_GT_174.png",
"f921a4fe-bbe2-4f29-90e7-40c024363840_GT_5842.png",
"d0828337-193e-4f57-8e90-a4a2e2947d9d_GT_19911.png",
"d9715e62-092c-4517-ba57-8b7eed6e18a9_GT_1637.png",
"b2677b41-7cea-4181-ab09-43f60bf5ebc1_GT_2334.png",
"2684e679-e335-4563-aef8-5f9c5000dd18_GT_990.png",
"d0f107bd-867c-4355-8e3b-945ff04446f0_GT_26530.png",
"d69d8dda-722f-4668-962b-0f9fd58a09b2_GT_312.png",
"b0c666e9-7ca4-4227-ae0b-f80bee479e69_GT_1393.png",
"bcbd1d1e-cdbb-4958-8a88-7e76578d0146_GT_4568.png",
"bcbd1d1e-cdbb-4958-8a88-7e76578d0146_GT_4290.png",
"a43a1946-ab5a-46ab-898e-6bb0b1c32cc9_GT_33757.png",
"ed351457-849e-4e27-9c6e-448718e9c4ac_GT_104547.png",
"c7d6c79f-924a-4231-a3bf-303b66d30f2c_GT_938.png",
"d0b3f9f3-7887-4b0c-91d1-14c67535019f_GT_1198.png",
"d69d8dda-722f-4668-962b-0f9fd58a09b2_GT_136.png",
"ed351457-849e-4e27-9c6e-448718e9c4ac_GT_104433.png",
"ff948169-f061-4d13-8f17-dddd42489350_GT_1913.png",
"ff948169-f061-4d13-8f17-dddd42489350_GT_2124.png",
"d789f889-f40d-4f4f-879a-faf8d1d5fbb5_GT_653.png",
"a9ed416e-7bb6-45bb-ad59-4205aee0c7d2_GT_358.png",
"3224744e-98a3-4b6d-8ddf-c9c457f369c6_GT_3882.png",
"bc5a7acf-43c1-40a4-82ea-56318b78f5f6_GT_12829.png",
"e54ecd80-4a03-4098-85ff-d8986cd93182_GT_525.png",
"0be77cff-5c73-4c4a-b788-c36c0b7616c1_GT_9961.png",
"a9ed416e-7bb6-45bb-ad59-4205aee0c7d2_GT_62.png",
"0be77cff-5c73-4c4a-b788-c36c0b7616c1_GT_10036.png",
"ac7da88d-9248-43ed-9f46-4965bfc01c0c_GT_1923.png",
"d5dc2fd1-e43c-4040-9218-6be98cd1158e_GT_3994.png",
"eb0fe763-7090-4132-b49f-9417b920195f_GT_32.png",
"be278a80-e70a-42da-8e0a-b4e36732bf16_GT_8800.png",
"3320384c-03dd-49d4-898c-e35f10eeb122_GT_1353.png",
"f50d66cf-6481-4f71-963c-fdae3e2da7f8_GT_552.png",
"b2677b41-7cea-4181-ab09-43f60bf5ebc1_GT_781.png",
"b0c666e9-7ca4-4227-ae0b-f80bee479e69_GT_1694.png",
"e54ecd80-4a03-4098-85ff-d8986cd93182_GT_751.png",
"0f6febd6-2913-477a-872b-6fe5dc0a95de_GT_12897.png",
"114d4887-8177-4682-ace8-5275ba137757_GT_14946.png",
"a43a1946-ab5a-46ab-898e-6bb0b1c32cc9_GT_34176.png",
"e783404f-9169-4011-be6d-6f729282da43_GT_603.png",
"d0f107bd-867c-4355-8e3b-945ff04446f0_GT_26431.png",
"3a0d2b4c-52ca-4212-a109-b2309241f783_GT_914.png",
"efccce8a-7bae-46d4-8e1e-8595931756c3_GT_2050.png",
"f69085ca-df9a-497f-a80a-236764b99c46_GT_43605.png",
"33c32af3-b069-43d3-85a6-939e3db2976a_GT_2666.png",
"b2677b41-7cea-4181-ab09-43f60bf5ebc1_GT_95.png",
"d0828337-193e-4f57-8e90-a4a2e2947d9d_GT_19858.png",
"d69d8dda-722f-4668-962b-0f9fd58a09b2_GT_878.png",
"4f9143ad-e811-4351-b888-c6445e5c26a4_GT_34931.png",
"ac7da88d-9248-43ed-9f46-4965bfc01c0c_GT_3790.png",
"4f9143ad-e811-4351-b888-c6445e5c26a4_GT_34762.png",
"fb60ca72-0a2b-45de-ad8a-f99cb6dd4910_GT_3451.png",
"be278a80-e70a-42da-8e0a-b4e36732bf16_GT_9094.png",
"d9715e62-092c-4517-ba57-8b7eed6e18a9_GT_3280.png",
"e93674d6-6420-4434-86a8-cb3837fc45b5_GT_5475.png",
"1f5a5a29-a951-4c68-ba31-b6aefeb96c18_GT_1176.png",
"b2677b41-7cea-4181-ab09-43f60bf5ebc1_GT_2052.png",
"f1eec0c6-b578-4dd3-9972-3fee1d22ad95_GT_10060.png",
"efccce8a-7bae-46d4-8e1e-8595931756c3_GT_2649.png",
"a43a1946-ab5a-46ab-898e-6bb0b1c32cc9_GT_34507.png",
"bb9ecc23-47d9-4b69-abf4-fd330eb2a293_GT_1667.png",
"e93674d6-6420-4434-86a8-cb3837fc45b5_GT_5128.png",
"e54ecd80-4a03-4098-85ff-d8986cd93182_GT_44.png",
"d8ac7e04-8e79-4f54-8f12-ee8d0ec08628_GT_5528.png",
"a43a1946-ab5a-46ab-898e-6bb0b1c32cc9_GT_33376.png",
"f69085ca-df9a-497f-a80a-236764b99c46_GT_44500.png",
"bb9ecc23-47d9-4b69-abf4-fd330eb2a293_GT_964.png",
"c7d6c79f-924a-4231-a3bf-303b66d30f2c_GT_151.png",
"d9715e62-092c-4517-ba57-8b7eed6e18a9_GT_2559.png",
"f921a4fe-bbe2-4f29-90e7-40c024363840_GT_5899.png",
"d69d8dda-722f-4668-962b-0f9fd58a09b2_GT_926.png",
"a43a1946-ab5a-46ab-898e-6bb0b1c32cc9_GT_34047.png",
"e783404f-9169-4011-be6d-6f729282da43_GT_464.png",
"e1de251d-e9b5-4e4a-9be0-c17b4cdc4f11_GT_1278.png",
"2684e679-e335-4563-aef8-5f9c5000dd18_GT_1159.png",
"d9715e62-092c-4517-ba57-8b7eed6e18a9_GT_1738.png",
"25a6b037-4436-4ce0-a9e1-a82a37a32b05_GT_52864.png",
"a43a1946-ab5a-46ab-898e-6bb0b1c32cc9_GT_33091.png",
"2684e679-e335-4563-aef8-5f9c5000dd18_GT_82.png",
"d91b9af2-f5ad-4448-9d3f-6d4b20c8203c_GT_94.png",
"e93674d6-6420-4434-86a8-cb3837fc45b5_GT_5579.png",
"f8a684b8-4991-48e3-8c74-77371485cac0_GT_325.png",
"c7d6c79f-924a-4231-a3bf-303b66d30f2c_GT_36.png",
"d823a7a2-cea9-45b9-9c3c-d5309b16fdf5_GT_348.png",
"b90cfc2f-7811-4d91-a929-7843781a2edb_GT_21314.png",
"bc5a7acf-43c1-40a4-82ea-56318b78f5f6_GT_12209.png",
"a0a71e0a-9ad2-4263-b8fc-8381d4efabc0_GT_1882.png",
"b0c666e9-7ca4-4227-ae0b-f80bee479e69_GT_994.png",
"a43a1946-ab5a-46ab-898e-6bb0b1c32cc9_GT_33930.png",
"ac7da88d-9248-43ed-9f46-4965bfc01c0c_GT_3701.png",
"e783404f-9169-4011-be6d-6f729282da43_GT_699.png",
"fbbba64e-127d-4eb1-a0d3-77af1baa230e_GT_1192.png",
"fbbba64e-127d-4eb1-a0d3-77af1baa230e_GT_1432.png",
"d823a7a2-cea9-45b9-9c3c-d5309b16fdf5_GT_242.png",
"d0828337-193e-4f57-8e90-a4a2e2947d9d_GT_19407.png",
"33c32af3-b069-43d3-85a6-939e3db2976a_GT_2725.png",
"259402d0-a583-4a45-98e3-b9c95878b436_GT_38222.png",
"c7d6c79f-924a-4231-a3bf-303b66d30f2c_GT_584.png",
"b90cfc2f-7811-4d91-a929-7843781a2edb_GT_21702.png",
"a9ed416e-7bb6-45bb-ad59-4205aee0c7d2_GT_323.png",
"1f3041c2-2971-49eb-95e0-ced6f46e7b6e_GT_22605.png",
"bb9ecc23-47d9-4b69-abf4-fd330eb2a293_GT_1104.png",
"3a0d2b4c-52ca-4212-a109-b2309241f783_GT_227.png",
"ea3124fb-18b6-4cd4-b610-e8c844f6820b_GT_27.png",
"bcbd1d1e-cdbb-4958-8a88-7e76578d0146_GT_4247.png",
"1d1747e8-1d23-4108-98da-4aa158386c54_GT_7027.png",
"d9715e62-092c-4517-ba57-8b7eed6e18a9_GT_1207.png",
"caa6110f-f2a3-410e-8db8-93bf43665a56_GT_496.png",
"0be77cff-5c73-4c4a-b788-c36c0b7616c1_GT_9416.png",
"b0988c4a-3373-472c-8990-db2f6729d100_GT_21458.png",
"3224744e-98a3-4b6d-8ddf-c9c457f369c6_GT_3782.png",
"d0b3f9f3-7887-4b0c-91d1-14c67535019f_GT_1061.png",
"bc5a7acf-43c1-40a4-82ea-56318b78f5f6_GT_12477.png",
"990e7b14-daa3-4685-ae55-6c4bd2c942a6_GT_297.png",
"8b6d4762-2c7d-49f5-94d0-3770513aa2c6_GT_33547.png",
"e93674d6-6420-4434-86a8-cb3837fc45b5_GT_4069.png",
"f821e60c-62d2-485c-8720-ad9c8530ba34_GT_16621.png",
"d91b9af2-f5ad-4448-9d3f-6d4b20c8203c_GT_107.png",
"bc5a7acf-43c1-40a4-82ea-56318b78f5f6_GT_11716.png",
"f69085ca-df9a-497f-a80a-236764b99c46_GT_44405.png",
"f1aa290e-0de4-4ccf-9e85-378d13b541c1_GT_35392.png",
"be278a80-e70a-42da-8e0a-b4e36732bf16_GT_8915.png",
"cceffabf-1649-4e65-80cc-be9aa2db8af4_GT_234.png",
"904bb444-c07a-4b1a-bcc5-083b7d26de4e_GT_9599.png",
"b0c666e9-7ca4-4227-ae0b-f80bee479e69_GT_440.png",
"bdb78738-396e-43a8-971f-e05635528dde_GT_25443.png",
"f8a684b8-4991-48e3-8c74-77371485cac0_GT_145.png",
"a9a70fdf-50b6-46f0-8b5a-e24cb90d8f74_GT_30072.png",
"1c3ea3e4-9a71-4c72-b5d0-d6f40f9eece6_GT_162.png",
"ff0478b5-5f2c-4450-8b90-d7450325e351_GT_552.png",
"b13bcaf6-f79c-4880-87cd-c7f1e8c72341_GT_6256.png",
"be278a80-e70a-42da-8e0a-b4e36732bf16_GT_8966.png",
"efccce8a-7bae-46d4-8e1e-8595931756c3_GT_1659.png",
"1c2d47b3-2cd5-4d48-8027-859eb5b9703d_GT_4863.png",
"d69d8dda-722f-4668-962b-0f9fd58a09b2_GT_447.png",
"d9715e62-092c-4517-ba57-8b7eed6e18a9_GT_1851.png",
"b2677b41-7cea-4181-ab09-43f60bf5ebc1_GT_168.png",
"b2677b41-7cea-4181-ab09-43f60bf5ebc1_GT_2239.png",
"d0f107bd-867c-4355-8e3b-945ff04446f0_GT_27182.png",
"a43a1946-ab5a-46ab-898e-6bb0b1c32cc9_GT_33295.png",
"a6d85655-1fdf-420a-ac58-36aed2696bec_GT_531.png",
"ec7ddfa9-5f5f-47e1-aadb-ad0222ae6230_GT_160.png",
"e1de251d-e9b5-4e4a-9be0-c17b4cdc4f11_GT_1265.png",
"34259f85-ded8-4748-8c7c-600383a53c4c_GT_14149.png",
"3fc66c96-e402-4e24-9e61-227be00c1d6c_GT_27873.png",
"f1eec0c6-b578-4dd3-9972-3fee1d22ad95_GT_9800.png",
"3a0d2b4c-52ca-4212-a109-b2309241f783_GT_1492.png",
"fb60ca72-0a2b-45de-ad8a-f99cb6dd4910_GT_3034.png",
"f6097e3c-821f-4602-abd2-e927adae4f41_GT_14463.png",
"caa6110f-f2a3-410e-8db8-93bf43665a56_GT_203.png",
"1f5a5a29-a951-4c68-ba31-b6aefeb96c18_GT_966.png",
"f50d66cf-6481-4f71-963c-fdae3e2da7f8_GT_228.png",
"0c5367f2-b43c-4af3-9751-0219c48796e9_GT_94.png",
"fbbba64e-127d-4eb1-a0d3-77af1baa230e_GT_498.png",
"a847cec6-5bac-4148-a717-7cc5f20d1cd6_GT_233.png",
"f793c363-b270-4e47-b8c6-03752a9b56f6_GT_1592.png",
"f69085ca-df9a-497f-a80a-236764b99c46_GT_43164.png",
"25a6b037-4436-4ce0-a9e1-a82a37a32b05_GT_52787.png",
"be9e2604-0fa0-4a33-88b5-ae7a3d03c996_GT_23710.png",
"e783404f-9169-4011-be6d-6f729282da43_GT_121.png",
"0c5367f2-b43c-4af3-9751-0219c48796e9_GT_341.png",
"b8045794-6524-42a6-a01a-07924bf3f32a_GT_4155.png",
"d37a25db-fa3f-4144-9a61-45d920a2f2e7_GT_97791.png",
"d0f107bd-867c-4355-8e3b-945ff04446f0_GT_25965.png",
"ff948169-f061-4d13-8f17-dddd42489350_GT_1103.png",
"25a6b037-4436-4ce0-a9e1-a82a37a32b05_GT_51855.png",
"a0a71e0a-9ad2-4263-b8fc-8381d4efabc0_GT_1236.png",
"114d4887-8177-4682-ace8-5275ba137757_GT_14214.png",
"f69085ca-df9a-497f-a80a-236764b99c46_GT_44003.png",
"e54ecd80-4a03-4098-85ff-d8986cd93182_GT_418.png",
"ac7da88d-9248-43ed-9f46-4965bfc01c0c_GT_719.png",
"ed351457-849e-4e27-9c6e-448718e9c4ac_GT_104379.png"]

    for seq_idx, seq in enumerate(dataloader['test']):

        seq_name = seq['meta']['paths'][0][0].split('/')[-3]
        print(seq_name)

        os.makedirs(os.path.join(args.out, seq_name, 'masks'), exist_ok=True)
        os.makedirs(os.path.join(args.out, seq_name, 'images'), exist_ok=True)
        os.makedirs(os.path.join(args.out, seq_name, 'flows'), exist_ok=True)
        results = open(os.path.join(args.out, seq_name, 'results.txt'), 'w')

        for frame in tqdm(range(args.prev_frames, len(seq['meta']['paths']))):

            path = [x.split('/')[-3] + '_GT_' + x.split('/')[-1][:-4] +'.png' for x in seq['meta']['paths'][frame]]
            # Load the data and mount them on cuda
            if frame == args.prev_frames:
                frames = [seq['image'][i].cuda(non_blocking=True) for i in range(frame + 1)]
                m2 = [torch.zeros((frames[0].shape[0], 2, 200, 200), device='cuda'),
                      torch.zeros((frames[0].shape[0], 2, 400, 400), device='cuda'),
                      torch.zeros((frames[0].shape[0], 2, 800, 800), device='cuda')]
                d2 = [F.interpolate(seq['image'][0], scale_factor=0.25).to(args.device),
                      F.interpolate(seq['image'][0], scale_factor=0.5).to(args.device),
                      seq['image'][0].to(args.device)]
            else:
                frames.append(seq['image'][frame].cuda(non_blocking=True))
                frames.pop(0)

            gt_dict = {task: seq[task][frame].cuda(non_blocking=True) if type(seq[task][frame]) is torch.Tensor else
            [e.cuda(non_blocking=True) for e in seq[task][frame]] for task in tasks}

            with torch.no_grad():
                outputs = model(frames[0], frames[1], m2, d2)
            outputs = dict(zip(tasks, outputs))

            name = seq['meta']['paths'][frame][0].split('/')[-1]
            save_outputs(args, seq_name, name, outputs)
            m2 = outputs['segment']
            d2 = outputs['deblur']

            task_metrics = {task: metrics_dict[task](outputs[task], gt_dict[task]) for task in tasks}
            metrics_values = {k: torch.round((10**3 * v))/(10**3) for task in tasks for k, v in task_metrics[task].items()}


            for metric in metrics:
                metric_cumltive[metric].append(metrics_values[metric])
                if path[0] in l:
                    metrics_hl[metric].append(metrics_values[metric])

            results.write(
                '\nFrame {}, PSNR: {:.3f},  SSIM: {:.3f}, EPE: {:.3f}, IoU: {:.3f} DiCE: {:.3f} IoU_HL: - DiCE_HL: -\n'.format(
                    name, metrics_values['psnr'], metrics_values['ssim'],
                    metrics_values['EPE'],
                    metrics_values['iou'], metrics_values['dice']))

        results.close()

    metric_averages = {m: sum(metric_cumltive[m])/len(metric_cumltive[m]) for m in metrics}
    metric_hl_averages = {m: sum(metrics_hl[m]) / len(metrics_hl[m]) for m in metrics}

    print("\n[TEST] {}\n".format(' '.join(['{}: {:.3f}'.format(m, metric_averages[m]) for m in metrics])))
    print("\n[TEST-HL] {}\n".format(' '.join(['{}: {:.3f}'.format(m, metric_hl_averages[m]) for m in metrics])))

    wandb_logs = {"Test - {}".format(m): metric_averages[m] for m in metrics}
    wandb_hl_logs = {"Test - HL - {}".format(m): metric_hl_averages[m] for m in metrics}
    wandb.log(wandb_logs)
    wandb.log(wandb_hl_logs)


def main(args):

    os.environ['CUDA_VISIBLE_DEVICES'] = "1"


    tasks = [task for task in ['segment', 'deblur', 'flow'] if getattr(args, task)]

    transformations = {'test': transforms.Compose([ToTensor(), Normalize()])}
    data = {'test': MTL_TestDataset(tasks, args.data_path, 'test', args.seq_len, transform=transformations['test'])}
    loader = {'test': DataLoader(data['test'], batch_size=args.bs, shuffle=False, num_workers=1, pin_memory=False)}


    metrics_dict = {
        'segment': SegmentationMetrics().to(args.device),
        'deblur': DeblurringMetrics().to(args.device),
        'flow': OpticalFlowMetrics().to(args.device)

    }
    metrics_dict = {k: v for k, v in metrics_dict.items() if k in tasks}


    model = VideoMIMOUNet(args, tasks, nr_blocks=args.nr_blocks, block=args.block).to(args.device)
    model = torch.nn.DataParallel(model).to(args.device)
    # Load checkpoint
    resume_path = os.path.join(args.out, 'ckpt_{}.pth'.format(70))
    state_dict = torch.load(resume_path)['state']
    model.load_state_dict(state_dict, strict=True)

    wandb.init(project='mtl-normal', entity='dst-cv', mode='disabled')
    wandb.run.name = args.out.split('/')[-1]
    wandb.watch(model)

    evaluate(args, loader, model, metrics_dict)


if __name__ == '__main__':
    parser = ArgumentParser(description='Parser of Training Arguments')

    parser.add_argument('--data', dest='data_path', help='Set dataset root_path', default='/media/efklidis/4TB/dblab_ecai', type=str) #/media/efklidis/4TB/ # ../raid/data_ours_new_split
    parser.add_argument('--out', dest='out', help='Set output path', default='/media/efklidis/4TB/RESULTS-ECAI/mostnet-sw/', type=str)
    parser.add_argument('--block', dest='block', help='Type of block "fft", "res", "inverted", "inverted_fft" ', default='res', type=str)
    parser.add_argument('--nr_blocks', dest='nr_blocks', help='Number of blocks', default=4, type=int)
    parser.add_argument("--device", dest='device', default="cuda", type=str)

    parser.add_argument("--segment", action='store_false', help="Flag for segmentation")
    parser.add_argument("--deblur", action='store_false', help="Flag for  deblurring")
    parser.add_argument("--flow", action='store_false', help="Flag for  optical flow")
    parser.add_argument("--resume", action='store_true', help="Flag for resume training")

    parser.add_argument('--bs', help='Set size of the batch size', default=1, type=int)
    parser.add_argument('--seq_len', dest='seq_len', help='Set length of the sequence', default=None, type=int)
    parser.add_argument('--prev_frames', dest='prev_frames', help='Set number of previous frames', default=1, type=int)

    parser.add_argument('--save_every', help='Save model every n epochs', default=1, type=int)


    args = parser.parse_args()

    print("> Parameters:")
    for p, v in zip(args.__dict__.keys(), args.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    main(args)
