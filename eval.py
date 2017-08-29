import numpy as np

def eval(val_txt, result_txt):

    iou_thresh = 0.4
    fp = []
    tp = []
    npos = 0

    val_f = open(val_txt)
    val_path = []
    val_result = []
    for line in val_f:
        if line.strip() == '':
            continue
        tmp_path = line.strip().replace('images', 'labels').replace('JPEGImages', 'labels')
        tmp_path = tmp_path.replace('.jpg', '.txt').replace('.png', '.txt')
        val_path.append(tmp_path)
        sub_f = open(tmp_path)
        for vline in sub_f:
            s = vline.split(' ')
            tmp_val = {'path':tmp_path, 'cls':int(s[0]), 'xx':float(s[1]), 'yy':float(s[2]), 'ww':float(s[3]), 'hh':float(s[4]), 'det':False}
            val_result.append(tmp_val)
            npos += 1

    result = []
    result_f = open(result_txt)
    for line in result_f:
        if line.strip() == '':
            continue
        s = line.strip().split(',')
        tmp_result = {'path':s[0], 'cls':int(s[1]), 'conf':float(s[2]), 'xx':float(s[3]), 'yy':float(s[4]),
                      'ww':float(s[5]), 'hh':float(s[6]), 'width':int(s[7]), 'height':int(s[8])}
        result.append(tmp_result)
        fp.append(0)
        tp.append(0)

    result.sort(key=lambda k: (k.get('conf', 0)), reverse=True)

    for i in range(len(result)):
        path = result[i]['path']
        x1 = float((result[i]['xx'] - result[i]['ww'] / 2) * result[i]['width'])
        y1 = float((result[i]['yy'] - result[i]['hh'] / 2) * result[i]['height'])
        x2 = float((result[i]['xx'] + result[i]['ww'] / 2) * result[i]['width'])
        y2 = float((result[i]['yy'] + result[i]['hh'] / 2) * result[i]['height'])
        bb = [x1, y1, x2, y2]
        val_list = [x for x in val_result if x['path'] == path]
        iou_max = 0
        j_max = 0
        for j in range(len(val_list)):
            if result[i]['cls'] == val_list[j]['cls']:
                x1gt = float((val_list[j]['xx'] - val_list[j]['ww'] / 2) * result[i]['width'])
                y1gt = float((val_list[j]['yy'] - val_list[j]['hh'] / 2) * result[i]['height'])
                x2gt = float((val_list[j]['xx'] + val_list[j]['ww'] / 2) * result[i]['width'])
                y2gt = float((val_list[j]['yy'] + val_list[j]['hh'] / 2) * result[i]['height'])
                bbgt = [x1gt, y1gt, x2gt, y2gt]
                bi=[max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
                iw=bi[2]-bi[0]+1
                ih=bi[3]-bi[1]+1
                if iw>0 and ih>0:
                    ua=(bb[2]-bb[0]+1)*(bb[3]-bb[1]+1)+(bbgt[2]-bbgt[0]+1)*(bbgt[3]-bbgt[1]+1)-iw*ih
                    iou=iw*ih/ua
                    if iou > iou_max:
                        iou_max = iou
                        j_max = j

        if iou_max >= iou_thresh:
            if not val_list[j_max]['det']:
                tp[i] = 1
                val_list[j_max]['det'] = True
            else:
                fp[i] = 1
        else:
            fp[i] = 1

    fp = np.asarray(fp, dtype=float)
    tp = np.asarray(tp, dtype=float)
    fp = fp.cumsum()
    tp = tp.cumsum()
    rec = tp / npos
    prec = tp / (fp+tp)

    ap=0
    for t in range(11):
        thresh = 0.1*t
        value = prec[rec >= thresh]
        if len(value) > 0:
            p = max(value)
            ap=ap+p/11

    return rec, prec, ap
