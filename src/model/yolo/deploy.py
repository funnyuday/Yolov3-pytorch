import pdb
import xlwt
import time
import torch
import numpy as np

from .utils import *
from torchvision import transforms
from torch.autograd import Variable


def deploy_yolo(opt, out, class_names):
    date = opt.deploy_time
    save_file = opt.accu_detail_file
    book = xlwt.Workbook(encoding="utf-8", style_compression=0)
    sheet = book.add_sheet("evaluation_resultm", cell_overwrite_ok=True)
    eval_res = ["Precision", "Recall", "AP", "F1-score", "log-MR"]

    alignment = xlwt.Alignment()
    alignment.horz = xlwt.Alignment.HORZ_CENTER
    alignment.vert = xlwt.Alignment.VERT_CENTER

    font = xlwt.Font()
    font.blod = True

    pattern_total_accu = xlwt.Pattern()
    pattern_total_accu.pattern = xlwt.Pattern.SOLID_PATTERN
    pattern_total_accu.pattern_fore_colour = 0x2B

    weights_color = xlwt.Pattern()
    weights_color.pattern = xlwt.Pattern.SOLID_PATTERN
    weights_color.pattern_fore_colour = 0x2C

    borders = xlwt.Borders()
    borders.left = 2
    borders.right = 2
    borders.top = 2
    borders.bottom = 2
    borders.bottom_colour = 0x3A

    style1 = xlwt.XFStyle()
    style1.alignment = alignment
    style1.font = font
    style1.borders = borders

    style2 = xlwt.XFStyle()
    style2.alignment = alignment
    style2.pattern = pattern_total_accu
    style2.font = font
    style2.borders = borders

    style3 = xlwt.XFStyle()
    style3.alignment = alignment
    style3.pattern = weights_color
    style3.font = font
    style3.borders = borders

    sheet.col(0).width = 256 * 15
    sheet.write_merge(0, len(out[0]), 0, 0, date, style1)

    sheet.col(1).width = 256 * 20
    test_inf = u'测试路径：' + opt.data_dir + '\n'
    sheet.write_merge(0, len(out[0]), 1, 1, test_inf, style1)

    sheet.col(2).width = 256 * 15
    sheet.write_merge(0, len(out[0]), 2, 2, opt.name, style3)

    sheet.col(3).width = 256 * 15
    sheet.write(0, 3, u'', style2)
    for i in range(len(out[0])):
        sheet.write(i + 1, 3, class_names[i], style2)
    
    first_col = 4
    for index, val in enumerate(eval_res):
        sheet.write_merge(0, 0, first_col, first_col + 1, val, style1)
        for col, write_val in enumerate(out[index]):
            sheet.write_merge(col + 1, col + 1, first_col, first_col + 1, "%.2f" % (write_val*100) + '%', style1)
        first_col += 2
    book.save(save_file)

def eval_yolo(opt, data_loader, test_model):
    model = test_model
    model.eval()
    opt.deploy_time = ".".join([str(x) for x in time.localtime()[:3]])
    class_names = load_classes(opt.names)
    labels = []
    sample_metrics = []
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    img_detections = []
    img_paths = []

    for i, data in data_loader:
        inputs, targets, img_path = data
        labels += targets[:, 1].tolist()
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, [2,4]] *= opt.img_size[0]
        targets[:, [3,5]] *= opt.img_size[1]
        
        imgs = Variable((inputs.float()).type(Tensor), requires_grad=False).cuda(opt.gpu_id)
        with torch.no_grad():
            outs = model(imgs)[1].detach().cpu()
            outs = non_max_suppression(outs, conf_thr=opt.conf_thr, iou_thr=opt.nms_thr)
        img_detections.extend(outs)
        img_paths.extend(img_path)
        sample_metrics += get_batch_statistics(outs, targets, iou_thr=opt.iou_thr)

    if len(sample_metrics) == 0:
        print("---- No detections over whole validation set ----")
        return None
    
    true_pos, pred_s, pred_l, iou_s = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    torch.cuda.empty_cache()

    draw_scatter_diagram(opt, pred_s, pred_l, iou_s, class_names)
    metrics_output = ap_per_class(true_pos, pred_s, pred_l, labels)
    deploy_yolo(opt, metrics_output, class_names)
    print_eval_stats(metrics_output, class_names)
    if opt.draw_res:
        draw_and_save_output_images(img_detections, img_paths, opt, class_names)


def predict_images(opt, test_model, image, colors, class_names):
    model = test_model
    model.eval()
    shape = image.shape
    ori_img = image
    image = transforms.ToTensor()(letter_box(opt.img_size, image))
    image = transforms.Normalize(std=[255.,255.,255.], mean=[1.,1.,1.])(image)
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    image = Variable((image.float()).type(Tensor), requires_grad=False).cuda(opt.gpu_id)
    rects = []
    with torch.no_grad():
        outs = model(image.unsqueeze(dim=0))[1].detach().cpu()
        outs = non_max_suppression(outs, conf_thr=opt.conf_thr, iou_thr=opt.nms_thr)
    for detections in outs:
        detections = rescale_boxes(detections, opt.img_size, shape[:2])
        for i, res in enumerate(detections):
            x1, y1, x2, y2, conf, cls_pred = res
            pred_name = class_names[int(cls_pred)]
            if pred_name not in opt.show_names:
                continue
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            name_color = colors[pred_name]
            ori_img = cv2.rectangle(ori_img, (x1, y1), (x2, y2), name_color, 2)
            text = "{}:{:.1f}%".format(pred_name, conf*100)
            txt_color = (0, 0, 0) if np.mean(name_color) > 0.5 else (255, 255, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX
            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
            txt_bk_color = name_color
            cv2.rectangle(ori_img, (x1, y1 + 1), (x1 + txt_size[0] + 1, y1 + int(1.5*txt_size[1])), txt_bk_color, -1)
            cv2.putText(ori_img, text, (x1, y1 + txt_size[1]), font, 0.4, txt_color, thickness=1)
            if opt.gen_txt:
                rects.append(xyxy2darknet(x1, y1, x2, y2, str(opt.show_names.index(pred_name)), ori_img.shape[:2]))
            
    return cv2.cvtColor(ori_img, cv2.COLOR_RGB2BGR), rects
        