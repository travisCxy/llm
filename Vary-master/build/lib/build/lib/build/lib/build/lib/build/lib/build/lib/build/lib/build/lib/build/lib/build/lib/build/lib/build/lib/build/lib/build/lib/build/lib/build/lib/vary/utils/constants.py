CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "log"

IGNORE_INDEX = -100
# DEFAULT_PAD_TOKEN = "[PAD]"

DEFAULT_PAD_TOKEN = "<|endoftext|>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_BOX_TOKEN = "<box>"

DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'

DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'

DEFAULT_BOXES_TOKEN = '<boxes>'
DEFAULT_AT_TOKEN = '<at>'


ROOT_PATH = '/data/public/ucaswei/data/'

CONVERSATION_DATA = {

    # pair 4m
    'laion-coco-4m': {
        'images': '',
        'annotations': '',
    }, 

    'cc665k': {
        'images': "/path_to/LLaVA1.5/images/",
        'annotations': "/path_to/LLaVA1.5/llava_v1_5_66k.json",
    },

    'pdf': {
        'images': "",
        'annotations': "",
    },

    'docvqa_train': {
        'images': "",
        'annotations': "",
    },

    'chartqa_train': {
        'images': "",
        'annotations': "",
    },

    'tiku0_train': {
        'images': "/mnt/ceph/tiku/images0/",
        'annotations': "/mnt/ceph/tiku/images0/conversations.json",
    },

    'tiku0-2_train': {
        'images': "/mnt/ceph/tiku/images0-2/",
        'annotations': "/mnt/ceph/tiku/images0-2/conversations.json",
    },

    'tiku1_train': {
        'images': "/mnt/ceph/tiku/images1/",
        'annotations': "/mnt/ceph/tiku/images1/conversations.json",
    },

    # 'tiku2_train': {
    #     'images': "/mnt/ceph/tiku/images2/",
    #     'annotations': "/mnt/ceph/tiku/images2/conversations.json",
    # },

    'box0_train': {
        'images': "/mnt/ceph/tiku/images_split/",
        'annotations': "/mnt/ceph/tiku/images_split/conversations_box_0_opt.json",
    },

    'box1_train': {
        'images': "/mnt/ceph/tiku/images_split/",
        'annotations': "/mnt/ceph/tiku/images_split/conversations_box_1_opt.json",
    },

    'tiku_html_box0_train': {
        'images': "/mnt/ceph/tiku/images_split/",
        'annotations': "/mnt/ceph/tiku/images_split/conversations_html_box_0.json",
    },

    'tiku_html_box1_train': {
        'images': "/mnt/ceph/tiku/images_split/",
        'annotations': "/mnt/ceph/tiku/images_split/conversations_html_box_1.json",
    },

    'sjb_box_train': {
        'images': "/mnt/ceph/tiku/sjb/",
        'annotations': "/mnt/ceph/tiku/sjb/conversations_box.json",
    },

    'sjb_html_box_train': {
        'images': "/mnt/ceph/tiku/sjb/",
        'annotations': "/mnt/ceph/tiku/sjb/conversations_html_box.json",
    },

    'sjb_html_cbox_train': {
        'images': "/mnt/ceph/tiku/sjb/",
        'annotations': "/mnt/ceph/tiku/sjb/conversations_html_cbox.json",
    },

    'sjb_html_nobox_train': {
        'images': "/mnt/ceph/tiku/sjb/",
        'annotations': "/mnt/ceph/tiku/sjb/conversations_html_nobox.json",
    },

    'tiku_html_box_train': {
        'images': "/mnt/ceph/tiku/images0/",
        'annotations': "/mnt/ceph/tiku/images0/conversations_html_box.json",
    },

    'tiku_split_html_box_train': {
        'images': "/mnt/ceph/tiku/images_split/",
        'annotations': "/mnt/ceph/tiku/images_split/conversations_html_box.json",
    },

    'det_html_box_train': {
        'images': "/mnt/ceph/15/datasets/yyt_det/20210602/YYT_DET_20210602/train/",
        'annotations': "/mnt/ceph/15/datasets/yyt_det/20210602/YYT_DET_20210602/train/conversations_box.json",
    },

    'tiku_box0_train': {
        'images': "/mnt/ceph/tiku/images_split/",
        'annotations': "/mnt/ceph/tiku/images_split/tiku_box_0.json",
    },

    'tiku_box1_train': {
        'images': "/mnt/ceph/tiku/images_split/",
        'annotations': "/mnt/ceph/tiku/images_split/tiku_box_1.json",
    },

    'tiku_next_box0_train': {
        'images': "/mnt/ceph/tiku/images_split/",
        'annotations': "/mnt/ceph/tiku/images_split/tiku_next_box_0.json",
    },

    'tiku_next_box1_train': {
        'images': "/mnt/ceph/tiku/images_split/",
        'annotations': "/mnt/ceph/tiku/images_split/tiku_next_box_1.json",
    },

    'sjb_html_train': {
        'images': "/mnt/ceph/tiku/sjb/",
        'annotations': "/mnt/ceph/tiku/sjb/conversations_html.json",
    },

    'tiku_html_train': {
        'images': "/mnt/ceph/tiku/",
        'annotations': "/mnt/ceph/tiku/sjb/conversations_tiku_html.json",
    },

    'sjb2_html_train': {
        'images': "/mnt/ceph/tiku/sjb2/",
        'annotations': "/mnt/ceph/tiku/sjb2/conversations_html.json",
    },

    'sjb2_tiku_html_train': {
        'images': "/mnt/ceph/tiku/sjb2/",
        'annotations': "/mnt/ceph/tiku/sjb2/conversations_tiku_html.json",
    },

    'mix_sjb_train': {
        'images': "/mnt/ceph2/datasets/tiku/mix/",
        'annotations': "/mnt/ceph2/datasets/tiku/mix/conversations_sjb.json",
    },

    'mix_tiku_train': {
        'images': "/mnt/ceph2/datasets/tiku/mix/",
        'annotations': "/mnt/ceph2/datasets/tiku/mix/conversations_tiku.json",
    },

    'tiku_box2_0_train': {
        'images': "/mnt/ceph/tiku/images_split/",
        'annotations': "/mnt/ceph/tiku/images_split/tiku_box2_0.json",
    },

    'tiku_box2_1_train': {
        'images': "/mnt/ceph/tiku/images_split/",
        'annotations': "/mnt/ceph/tiku/images_split/tiku_box2_1.json",
    },

    # 'det_train': {
    #     'images': "/mnt/ceph/15/datasets/yyt_det/20210602/YYT_DET_20210602/",
    #     'annotations': "/mnt/ceph/15/datasets/yyt_det/20210602/YYT_DET_20210602/train/conversations_det.json",
    # },
    #
    # 'qms_train': {
    #     'images': "/home/ateam/wu.changqing/qms/",
    #     'annotations': "/mnt/ceph/tiku/qms/conversations_qms.json",
    # },

    'mix_tiku_1M_train': {
        'images': "/mnt/ceph/tiku/",
        'annotations': "/mnt/ceph/tiku/mix/conversations_tiku_1M.json",
    },

    'mix_tiku_ocr_1M_train': {
        'images': "/mnt/ceph2/datasets/tiku/",
        'annotations': "/mnt/ceph2/datasets/tiku/mix/conversations_tiku_ocr_1M.json",
    },

    # 'det_train': {
    #     'images': "/mnt/ceph2/datasets/YYT_DET_20210602/",
    #     'annotations': "/mnt/ceph2/datasets/tiku/det/conversations_det.json",
    # },
    #
    # 'qms_train': {
    #     'images': "/mnt/ceph2/datasets/qms/",
    #     'annotations': "/mnt/ceph2/datasets/tiku/det/conversations_qms_5w.json",
    # },

    'det_train': {
        'images': "/mnt/ceph2/datasets/YYT_DET_20210602/",
        'annotations': "/mnt/ceph2/datasets/tiku/pretrain/conversations_det.json",
    },

    'qms_train': {
        'images': "/mnt/ceph2/datasets/qms/",
        'annotations': "/mnt/ceph2/datasets/tiku/pretrain/conversations_qms_5w.json",
    },

    # 'tiku_ocr_1M_train': {
    #     'images': "/mnt/ceph2/datasets/tiku/",
    #     'annotations': "/mnt/ceph2/datasets/tiku/pretrain/conversations_tiku_ocr_1M.json",
    # },

    'conversations_caption': {
        'images': "/mnt/ceph/datasets/coco/train2014/",
        'annotations': "/mnt/ceph2/datasets/tiku/pretrain/conversations_caption.json",
    },

    'conversations_rec': {
        'images': "/mnt/ceph/datasets/coco/train2014/",
        'annotations': "/mnt/ceph2/datasets/tiku/pretrain/conversations_rec.json",
    },

    'conversations_reg': {
        'images': "/mnt/ceph/datasets/coco/train2014/",
        'annotations': "/mnt/ceph2/datasets/tiku/pretrain/conversations_reg.json",
    },

    'sjb_train': {
        'images': "/mnt/ceph2/datasets/tiku/",
        'annotations': "/mnt/ceph2/datasets/tiku/vary/conversations_sjb.json",
    },

    'tiku_train': {
        'images': "/mnt/ceph2/datasets/tiku/",
        'annotations': "/mnt/ceph2/datasets/tiku/vary/conversations_tiku.json",
    },

    'data_train': {
        'images': "/mnt/ceph2/datasets/data_large/",
        'annotations': "/mnt/ceph2/datasets/data_large/conversations.json",
    },

    'sjb_point_train': {
        'images': "/mnt/ceph2/datasets/tiku/",
        'annotations': "/mnt/ceph2/datasets/tiku/vary/conversations_sjb_point.json",
    },

    'tiku_point_train': {
        'images': "/mnt/ceph2/datasets/tiku/",
        'annotations': "/mnt/ceph2/datasets/tiku/vary/conversations_tiku_point.json",
    },

    'det2_train': {
        'images': "/mnt/ceph2/datasets/YYT_DET_20210602/",
        'annotations': "/mnt/ceph2/datasets/tiku/pretrain/conversations_det2.json",
    },

    'qms2_train': {
        'images': "/mnt/ceph2/datasets/qms/",
        'annotations': "/mnt/ceph2/datasets/tiku/pretrain/conversations_qms2.json",
    },

    'tiku_html_1M_train': {
        'images': "/mnt/ceph2/datasets/tiku/",
        'annotations': "/mnt/ceph2/datasets/tiku/vary/conversations_tiku_html_1M.json",
    },

    'sjb2_train': {
        'images': "/mnt/ceph2/datasets/tiku/",
        'annotations': "/mnt/ceph2/datasets/tiku/vary/conversations_sjb2.json",
    },

    'tiku2_train': {
        'images': "/mnt/ceph2/datasets/tiku/",
        'annotations': "/mnt/ceph2/datasets/tiku/vary/conversations_tiku2.json",
    },

    'sjb_det_train': {
        'images': "/mnt/ceph2/datasets/tiku/",
        'annotations': "/mnt/ceph2/datasets/tiku/vary/conversations_sjb_det.json",
    },

    'tiku_det_train': {
        'images': "/mnt/ceph2/datasets/tiku/",
        'annotations': "/mnt/ceph2/datasets/tiku/vary/conversations_tiku_det.json",
    },

    'tiku_font_train': {
        'images': "/mnt/ceph2/datasets/tiku/",
        'annotations': "/mnt/ceph2/datasets/tiku/vary/conversations_tiku_font.json",
    },

    'tiku_font2_train': {
        'images': "/mnt/ceph2/datasets/tiku/",
        'annotations': "/mnt/ceph2/datasets/tiku/vary/conversations_tiku_font2.json",
    },

    'sjb_det2_train': {
        'images': "/mnt/ceph2/datasets/tiku/",
        'annotations': "/mnt/ceph2/datasets/tiku/vary/conversations_sjb_det2.json",
    },

    'tiku_det2_train': {
        'images': "/mnt/ceph2/datasets/tiku/",
        'annotations': "/mnt/ceph2/datasets/tiku/vary/conversations_tiku_det2.json",
    },

    'sjb_det3_train': {
        'images': "/mnt/ceph2/datasets/tiku/",
        'annotations': "/mnt/ceph2/datasets/tiku/vary/conversations_sjb_det3.json",
    },

    'tiku_det3_train': {
        'images': "/mnt/ceph2/datasets/tiku/",
        'annotations': "/mnt/ceph2/datasets/tiku/vary/conversations_tiku_det3.json",
    },

    'sjb_det4_train': {
        'images': "/mnt/ceph2/datasets/tiku/",
        'annotations': "/mnt/ceph2/datasets/tiku/vary/conversations_sjb_det4.json",
    },

    'tiku_det4_train': {
        'images': "/mnt/ceph2/datasets/tiku/",
        'annotations': "/mnt/ceph2/datasets/tiku/vary/conversations_tiku_det4.json",
    },

    'opt_det2_train': {
        'images': "/mnt/ceph2/datasets/YYT_DET_20210602/",
        'annotations': "/mnt/ceph2/datasets/tiku/opt/conversations_det2.json",
    },

    'opt_qms2_train': {
        'images': "/mnt/ceph2/datasets/qms/",
        'annotations': "/mnt/ceph2/datasets/tiku/opt/conversations_qms2.json",
    },

    'sjb_det5_train': {
        'images': "/mnt/ceph2/datasets/tiku/",
        'annotations': "/mnt/ceph2/datasets/tiku/vary/conversations_sjb_det5.json",
    },

    'tiku_det5_train': {
        'images': "/mnt/ceph2/datasets/tiku/",
        'annotations': "/mnt/ceph2/datasets/tiku/vary/conversations_tiku_det5.json",
    },

    'sjb_det6_train': {
        'images': "/mnt/ceph2/datasets/tiku/",
        'annotations': "/mnt/ceph2/datasets/tiku/vary/conversations_sjb_det6.json",
    },

    'tiku_det6_train': {
        'images': "/mnt/ceph2/datasets/tiku/",
        'annotations': "/mnt/ceph2/datasets/tiku/vary/conversations_tiku_det6.json",
    },

    'tiku_det7_train': {
        'images': "/mnt/ceph2/datasets/tiku3/",
        'annotations': "/mnt/ceph2/datasets/tiku/vary/conversations_tiku_det7.json",
    },

    'tiku_det8_train': {
        'images': "/mnt/ceph2/datasets/",
        'annotations': "/mnt/ceph2/datasets/tiku/vary/conversations_tiku_det8.json",
    },

    'sjb_det9_train': {
        'images': "/mnt/ceph2/datasets/",
        'annotations': "/mnt/ceph2/datasets/tiku/vary/conversations_sjb_det9.json",
    },

    'tiku_det9_train': {
        'images': "/mnt/ceph2/datasets/",
        'annotations': "/mnt/ceph2/datasets/tiku/vary/conversations_tiku_det9.json",
    },

    'opt_tiku_ocr_1M_train': {
        'images': "/mnt/ceph2/datasets/tiku/",
        'annotations': "/mnt/ceph2/datasets/tiku/opt/conversations_tiku_ocr_1M.json",
    },

    'tiku_ocr_1M_train': {
        'images': "/mnt/ceph2/datasets/tiku/",
        'annotations': "/mnt/ceph2/datasets/tiku/ocr/conversations_tiku_ocr_1M.json",
    },
}
