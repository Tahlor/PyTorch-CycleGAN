# Get HWR recognizer
import sys
sys.path.append("../simple_hwr")
import crnn
from pathlib import Path
import json
from hwr_utils import calculate_cer, Decoder
import cv2
import numpy as np
#from train import make_dataloaders
import torch
from torch.autograd import Variable

class HWR:
    def __init__(self, hwr_path=None, device="cuda", use_beam=False, optimizer=None, batch_size=1):
        if hwr_path is None:
            hwr_path = Path("/media/data/GitHub/simple_hwr/results/BEST/20190807_104745-smallv2/RESUME_model.pt")
            #hwr_path = Path("/media/data/GitHub/simple_hwr/results/BEST/LARGEv2/LARGE_model.pt")
        self.device = device
        self.model = self.load_hwr(hwr_path)

        self.decoder = Decoder(idx_to_char=self.idx_to_char, beam=use_beam)

        # Define CTC loss
        ctc = torch.nn.CTCLoss()
        log_softmax = torch.nn.LogSoftmax(dim=2).to(device)
        self.ctc_criterion = lambda x, y, z, t: ctc(log_softmax(x), y, z, t)

        self.optimizer = optimizer
        # config = {"training_jsons": , "char_to_idx":, "input_height":, "num_of_channels":, "training_warp":False}      
        self.online = torch.Tensor([0]).repeat(1,batch_size).view(1, -1, 1).to(device) if self.model.rnn.rnn.input_size != 1024 else None
        
    def load_hwr(self, hwr_path):
        hwr_model_dict = torch.load(str(hwr_path))
        model = hwr_model_dict["model_definition"]
        model.load_state_dict(hwr_model_dict['model'])
        self.idx_to_char = hwr_model_dict["idx_to_char"]
        self.char_to_idx = hwr_model_dict["char_to_idx"]
        return model.to(self.device) 

    def hwr_loss(self, image, label, label_lengths):
        self.model.train()
        labels = Variable(label, requires_grad=False)  # numeric indices version of ground truth
        label_lengths = Variable(label_lengths, requires_grad=False)
        pred_text = self.model(image, self.online)[0]
        preds_size = Variable(torch.IntTensor([pred_text.size(0)] * pred_text.size(1)))
        loss_recognizer = self.ctc_criterion(pred_text, labels, preds_size, label_lengths)
        
        # self.optimizer.zero_grad()
        # loss_recognizer.backward(retain_graph=False)
        # self.optimizer.step()

        # predicted = self.decoder.decode_training(pred_text.cpu().permute(1, 0, 2))
        return loss_recognizer

        
    def cer(self, pred, label):
        return calculate_cer(pred, label)
        
    def hwr_predict(self, image, as_string=True):
        self.model.eval()
        batch_output = self.model(image, self.online)[0].cpu().permute(1, 0, 2)
        if as_string:
            return " ".join(self.decoder.decode_training(batch_output, as_string=True))
        else:
            return list(self.decoder.decode_training(batch_output, as_string=False))

    def read_from_file(self, path):
        img = cv2.imread(test_image, 0)
        percent = float(60) / img.shape[0]
        img = cv2.resize(img, (0, 0), fx=percent, fy=percent, interpolation=cv2.INTER_CUBIC) / 128.0 - 1.0

        # Add a channel dimension
        img=img[np.newaxis, np.newaxis,:,:]
        return Variable(torch.FloatTensor(img).to(self.device), requires_grad=False)

def build_labels(labels, device="cpu"):
    out = []
    label_lengths = []
    for x in labels:
        out += x
        label_lengths.append(len(x))
    out = np.array(out).astype(int)
    #return torch.tensor(out).type(torch.cuda.IntTensor).to(device), torch.tensor(label_lengths).type(torch.cuda.IntTensor).to(device)
    return torch.tensor(out, requires_grad=False).to(device), torch.tensor(label_lengths, requires_grad=False).to(device)

if __name__=="__main__":
    test_image = r"/media/data/GitHub/PyTorch-CycleGAN/data/test_offline_preprocessed/m01-049-00.png"
    #test_image = r"/media/data/GitHub/PyTorch-CycleGAN/data/test_offline_preprocessed/m01-095-03.png"
    #test_image = r"/media/data/GitHub/PyTorch-CycleGAN/data/test_offline_preprocessed/m01-125-00.png"
    hw = HWR()
    img = hw.read_from_file(test_image)
    print(hw.hwr_predict(img))
