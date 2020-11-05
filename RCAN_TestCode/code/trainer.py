import os
import math
from decimal import Decimal

import utility

import torch
from torch.autograd import Variable
from tqdm import tqdm

import numpy as np
import cv2
from skimage import io
import skimage.measure

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

        self.error_last = 1e8

    def train(self):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (lr, hr, _, idx_scale) in enumerate(self.loader_train):
            lr, hr = self.prepare([lr, hr])
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr, idx_scale)
            loss = self.loss(sr, hr)
            if loss.item() < self.args.skip_threshold * self.error_last:
                loss.backward()
                self.optimizer.step()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

    def test(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        # self.model.eval()

        # 只加载模型参数
        path_state_dict = "/export/liuzhe/program2/RCAN_test/RCAN_TestCode/model/model_mid/epoch_10.pt"
        state_dict_load = torch.load(path_state_dict)
        self.model.load_state_dict(state_dict_load['net'], False)
        # self.model.load_state_dict(state_dict_load['net'], strict = False)
        # self.model.cuda()
        self.model.eval()

        timer_test = utility.timer()
        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                eval_acc = 0
                self.loader_test.dataset.set_scale(idx_scale)
                tqdm_test = tqdm(self.loader_test, ncols=80)
                for idx_img, (lr, hr, filename, hrimg, _) in enumerate(tqdm_test):
                    filename = filename[0]
                    no_eval = (hr.nelement() == 1)

                    # print(hrimg.shape)      # [1, 3, 512, 512]
                    # print(hr.shape)     # [1, 8, 512, 512]
                    # print(lr.shape)     # [1, 3, 256, 256]

                    if not no_eval:
                        lr, hr = self.prepare([lr, hr])
                    else:
                        lr = self.prepare([lr])[0]
                    from PIL import Image

                    # print(hrimg.shape)      # [1, 3, 512, 512]
                    # print("hr shape=", hr.shape)     # [1, 8, 512, 512]
                    # print(lr.shape)     # [1, 3, 256, 256]

                    # 保存原hr图像
                    # hrimg_1 = np.transpose(hrimg.cpu().detach().data.numpy()[0],(1,2,0))
                    # print(hrimg_1.shape)    # (512, 512, 3)
                    # io.imsave('/export/liuzhe/program2/RCAN_test/RCAN_TestCode/SR/BI/hrimg.png', hrimg_1)    #hr图像


                    # print(np.max(hrimg_1))   # 255
                    # img_cr, img_cb = self.get_hr_cbcr(hrimg)
                    img_cr, img_cb = self.get_hr_cbcr(hr)

                    sr = self.model(lr, idx_scale)
                    sr = self.channel_8_3(sr, img_cr, img_cb)
                    # sr1 = sr

                    # print("sr shape=", sr.shape)     # torch.Size([1, 8, 512, 512])
                    # print(type(sr))     # <class 'torch.Tensor'>

                    # 验证格雷码编码与解码，并保存图片
                    # hr_8_3 = self.channel_8_3(hr, img_cr, img_cb)   # torch.Size([1, 3, 512, 512])
                    # srimg = np.transpose(hr_8_3.cpu().detach().data.numpy()[0], (1, 2, 0))
                    # io.imsave('/export/liuzhe/program2/RCAN_test/RCAN_TestCode/SR/BI/srimg.png', srimg)    # hr格雷码、反格雷码之后的图像
                    # sr = self.channel_8_3(sr, img_cr, img_cb)  # torch.Size([1, 3, 512, 512])
                    # srimg1 = np.transpose(sr.cpu().detach().data.numpy()[0], (1, 2, 0))
                    # io.imsave('/export/liuzhe/program2/RCAN_test/RCAN_TestCode/SR/BI/srimg1.png', srimg1)  # lr格雷码经过超分网络、反格雷码之后的图像

                    # print(np.max(srimg))

                    # sr = utility.quantize(sr, self.args.rgb_range)

                    # 计算psnr结果
                    # psnr = skimage.measure.compare_psnr(srimg/255, srimg1/255, data_range=1)
                    # psnr = skimage.measure.compare_psnr(srimg1 / 255, hrimg_1 / 255, data_range=1)
                    # print("psnr= ", psnr)
                    # exit(-1)

                    save_list = [sr]
                    # 计算psnr
                    # if not no_eval:
                    #     eval_acc += utility.calc_psnr(
                    #         sr, hr, scale, self.args.rgb_range,
                    #         benchmark=self.loader_test.dataset.benchmark
                    #     )
                    #     save_list.extend([lr, hr])

                    if self.args.save_results:
                        #self.ckp.save_results(filename, save_list, scale)
                        self.ckp.save_results_nopostfix(filename, save_list, scale)

                self.ckp.log[-1, idx_scale] = eval_acc / len(self.loader_test)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        self.args.data_test,
                        scale,
                        self.ckp.log[-1, idx_scale],
                        best[0][idx_scale],
                        best[1][idx_scale] + 1
                    )
                )

        self.ckp.write_log(
            'Total time: {:.2f}s, ave time: {:.2f}s\n'.format(timer_test.toc(), timer_test.toc()/len(self.loader_test)), refresh=True
        )
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))

    def prepare(self, l, volatile=False):
        device = torch.device('cpu' if self.args.cpu else 'cuda')

        # print(self.args.precision)
        def _prepare(tensor):

            if self.args.precision == 'half': tensor = tensor.half()

            return tensor.to(device)
           
        return [_prepare(_l) for _l in l]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs

    def get_hr_cbcr(self, hr):

        # [1,3,512,512]
        x0 = (hr.cpu().detach().data.numpy()).astype("uint8")  # x转numpy

        x1 = np.zeros([x0.shape[0], x0.shape[2], x0.shape[3]], dtype='uint8')
        x2 = np.zeros([x0.shape[0], x0.shape[2], x0.shape[3]], dtype='uint8')
        x3 = np.zeros([x0.shape[0], x0.shape[2], x0.shape[3]], dtype='uint8')

        for i in range(x0.shape[0]):
            img = x0[i, :, :, :]
            img = np.transpose(img, (1, 2, 0))  # 转成[h,w,c]
            r, g, b = cv2.split(img)
            img = cv2.merge([b, g, r])
            # print(img.shape)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
            # img_y = img[:, :, 0]
            img_cr = img[:, :, 1]
            img_cb = img[:, :, 2]

            # x1[i, :, :] = img_y
            x2[i, :, :] = img_cr
            x3[i, :, :] = img_cb

        # print(x2.shape)
        return x2, x3

    def channel_8_3(self, sr, img_cr, img_cb):
        # [1, 8, 512, 512]
        x_0 = (sr.cpu().detach().data.numpy()).astype("uint8")  # x转numpy

        x_1 = np.zeros([x_0.shape[0], x_0.shape[2], x_0.shape[3]], dtype='uint8')   # [1, 512 ,512]

        # x_2 = np.zeros([3, x_0.shape[2], x_0.shape[3]], dtype = np.uint8)   # [3, h ,w]

        x_finally = np.zeros([x_0.shape[0], 3, x_0.shape[2], x_0.shape[3]], dtype='uint8')   # [1,3,512,512]

        for i in range(x_0.shape[0]):
            img1 = x_0[i, :, :, :]    # [8, 344 ,228]
            x_1[x_0.shape[0]-1, :, :] = img1[0, :, :]*1 + img1[1, :, :]*2 + img1[2, :, :]*4 + img1[3, :, :]*8 + img1[4, :, :]*16 + img1[5, :, :]*32 + img1[6, :, :]*64 + img1[7, :, :]*128

        img_1 = x_1     # [1, 512, 512]

        # 十进制格雷码转二进制的十进制数
        img_2 = (img_1.astype("uint8") / 2).astype("uint8")  # 除2取下界等于右移位运算
        img = img_1.astype("uint8")
        img_2 = img ^ img_2  # 异或运算   [1, 512, 512]

        x_finally[:, 0, :, :] = img_2
        x_finally[:, 1, :, :] = img_cr
        x_finally[:, 2, :, :] = img_cb      # x_finally.shape = [1,3,512,512]

        # ycrcb转rgb
        a = x_finally[0, :, :, :]   # [3, h, w]
        a = np.transpose(a, (1, 2, 0))  # 转成  [512, 512, 3]

        # y, cb, cr = cv2.split(img)
        # img = cv2.merge([y, cr, cb])
        img = cv2.cvtColor(a, cv2.COLOR_YCrCb2RGB)  # [512,512,3]
        # b,g,r = cv2.split(img)
        # img = cv2.merge([r,g,b])
        img = np.transpose(img, (2, 0, 1))  # 转成[3,512,512]
        for i in range(x_0.shape[0]):
            x_finally[i, :, :, :] = img

        x_finally = torch.from_numpy(np.ascontiguousarray(x_finally)).float()   # [1,3,512,512]
        x_finally = x_finally.cuda()
        # print(x_finally.shape)

        return x_finally


