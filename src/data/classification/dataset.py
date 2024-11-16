from PIL import Image
from torch.utils.data import Dataset
from ..data_utils import get_data, get_transformer

class RealFakeDataset(Dataset):
    def __init__(self, opt):
        assert opt.data_label in ["train", "val", "test"]
        self.data_label  = opt.data_label

        self.total_list, self.labels_dict = get_data(opt)
        self.transform = get_transformer(opt)

    def __len__(self):
        return len(self.total_list)


    def __getitem__(self, idx):
        img_path = self.total_list[idx]
        label = self.labels_dict[img_path]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, label


