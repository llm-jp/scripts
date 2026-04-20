import logging

logging.basicConfig(level=logging.INFO)

PRETRAIN_PHASE1_DATASET_ROOT = "/data/llmjp-test/experiments/llm-jp-pre-mid-training/pretrain/tokenized/v4.0alpha1.0/data"
PRETRAIN_PHASE2_DATASET_ROOT = "/data/llmjp-test/experiments/llm-jp-pre-mid-training/pretrain/tokenized/v4.0alpha1.0/data"
MIDTRAIN_PHASE1_DATASET_ROOT = "/data/llmjp-test/experiments/llm-jp-pre-mid-training/midtrain/tokenized/v4.0alpha1.0/data"

MIDTRAIN_DATASET_RATIO = 0.5
PRETRAIN_DATASET_RATIO = 1.0 - MIDTRAIN_DATASET_RATIO

# Dataset format:
# (num_repeats, num_tokens, dataset_path)

PRETRAIN_PHASE1_DATASET = [
    # Code datasets
    (8, 106882005818, f"{PRETRAIN_PHASE1_DATASET_ROOT}/code_olmo-starcoder_0000_text_document"),
    # NOTE(odashi): Stack v1 is replaced with Stack v2 in Phase2.
    # (0, 117852374347, f"{PRETRAIN_PHASE1_DATASET_ROOT}/code_stack_0000_text_document"),

    # English curated datasets
    (8, 5139896520, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_dolma-books_0000_text_document"),
    (8, 60036516798, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_dolma-pes2o_0000_text_document"),
    (8, 82254295911, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_dolma-reddit_0000_text_document"),
    (8, 3857521208, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_dolma-wiki_0000_text_document"),
    (8, 1483419806, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_dolmino-stackexchange_0000_text_document"),
    (8, 3141777, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_gsm8k_0000_text_document"),
    (8, 8750035604, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_mathpile_0000_text_document"),
    (8, 12977175126, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_olmo-algebraicstack_0000_text_document"),
    (8, 21716303067, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_olmo-arxiv_0000_text_document"),
    (8, 13171054142, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_olmo-openwebmath_0000_text_document"),
    (8, 4746637139, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_wiki_0000_text_document"),

    # English fineWeb low-scored
    (1, 102568520111, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_10_0000_text_document"),
    (1, 102509087783, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_10_0001_text_document"),
    (1, 100816401574, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_10_0002_text_document"),
    (1, 100065810915, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_10_0003_text_document"),
    (1, 99955083033, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_10_0004_text_document"),
    (1, 100268985585, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_10_0005_text_document"),
    (1, 98582941635, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_10_0006_text_document"),
    (1, 102501814684, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_10_0007_text_document"),
    (1, 87146714512, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_10_0008_text_document"),
    (1, 102540352684, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_11_0000_text_document"),
    (1, 101615714055, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_11_0001_text_document"),
    (1, 100223579527, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_11_0002_text_document"),
    (1, 99832954483, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_11_0003_text_document"),
    (1, 100200285660, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_11_0004_text_document"),
    (1, 98939237258, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_11_0005_text_document"),
    (1, 102361066324, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_11_0006_text_document"),
    (1, 100127948990, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_11_0007_text_document"),
    (1, 51572633747, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_11_0008_text_document"),
    (1, 102346538767, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_12_0000_text_document"),
    (1, 100658650543, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_12_0001_text_document"),
    (1, 99879930853, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_12_0002_text_document"),
    (1, 100207021051, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_12_0003_text_document"),
    (1, 99609557924, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_12_0004_text_document"),
    (1, 101835220299, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_12_0005_text_document"),
    (1, 99887438413, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_12_0006_text_document"),
    (1, 91670119172, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_12_0007_text_document"),
    (1, 101899332058, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_13_0000_text_document"),
    (1, 100107151548, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_13_0001_text_document"),
    (1, 100188263808, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_13_0002_text_document"),
    (1, 100208220130, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_13_0003_text_document"),
    (1, 101460435434, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_13_0004_text_document"),
    (1, 99804308223, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_13_0005_text_document"),
    (1, 99868561720, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_13_0006_text_document"),
    (1, 28118083173, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_13_0007_text_document"),
    (1, 101287815642, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_14_0000_text_document"),
    (1, 100193448611, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_14_0001_text_document"),
    (1, 100909877098, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_14_0002_text_document"),
    (1, 100757143238, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_14_0003_text_document"),
    (1, 99723340081, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_14_0004_text_document"),
    (1, 99265358219, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_14_0005_text_document"),
    (1, 15921206946, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_14_0006_text_document"),
    (1, 101005877270, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_15_0000_text_document"),
    (1, 100489515421, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_15_0001_text_document"),
    (1, 101723685894, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_15_0002_text_document"),
    (1, 100045434530, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_15_0003_text_document"),
    (1, 99720256993, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_15_0004_text_document"),
    (1, 98543462382, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_15_0005_text_document"),
    (1, 7338842460, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_15_0006_text_document"),
    (1, 100825885054, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_16_0000_text_document"),
    (1, 101739112228, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_16_0001_text_document"),
    (1, 100416142859, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_16_0002_text_document"),
    (1, 99754212843, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_16_0003_text_document"),
    (1, 100042039542, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_16_0004_text_document"),
    (1, 45275202852, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_16_0005_text_document"),
    (1, 101199070911, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_17_0000_text_document"),
    (1, 101438063034, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_17_0001_text_document"),
    (1, 99938292420, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_17_0002_text_document"),
    (1, 99892259073, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_17_0003_text_document"),
    (1, 88320720228, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_17_0004_text_document"),
    (1, 101682793329, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_18_0000_text_document"),
    (1, 100779083985, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_18_0001_text_document"),
    (1, 99807036301, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_18_0002_text_document"),
    (1, 100032616702, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_18_0003_text_document"),
    (1, 34415854865, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_18_0004_text_document"),
    (1, 101794562305, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_19_0000_text_document"),
    (1, 100159790313, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_19_0001_text_document"),
    (1, 100003883373, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_19_0002_text_document"),
    (1, 56293984847, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_19_0003_text_document"),

    # English fineWeb high-scored
    (4, 101909608393, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_20_0000_text_document"),
    (4, 100251912003, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_20_0001_text_document"),
    (4, 100024677537, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_20_0002_text_document"),
    (4, 77617886249, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_20_0003_text_document"),
    (4, 101494962972, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_21_0000_text_document"),
    (4, 99989476480, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_21_0001_text_document"),
    (4, 72610042095, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_21_0002_text_document"),
    (4, 101424615382, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_22_0000_text_document"),
    (4, 99977160848, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_22_0001_text_document"),
    (4, 76067058446, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_22_0002_text_document"),
    (4, 100987625784, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_23_0000_text_document"),
    (4, 99272893036, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_23_0001_text_document"),
    (4, 4407565031, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_23_0002_text_document"),
    (4, 100765663494, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_24_0000_text_document"),
    (4, 75847299078, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_24_0001_text_document"),
    (4, 100728857165, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_25_0000_text_document"),
    (4, 73395547895, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_25_0001_text_document"),
    (4, 100460689403, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_26_0000_text_document"),
    (4, 23787080600, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_26_0001_text_document"),
    (4, 100402654509, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_27_0000_text_document"),
    (4, 18288732636, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_27_0001_text_document"),
    (4, 81686513887, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_28_0000_text_document"),
    (4, 65145918651, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_29_0000_text_document"),
    (4, 100298736946, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_30_0000_text_document"),
    (4, 67634605260, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_fineweb-rescored_score_30_0001_text_document"),

    # Japanese curated datasets
    (8, 124537838, f"{PRETRAIN_PHASE1_DATASET_ROOT}/ja_aozorabunko_0000_text_document"),
    (8, 12476129929, f"{PRETRAIN_PHASE1_DATASET_ROOT}/ja_ceek-news_0000_text_document"),
    (8, 67690089, f"{PRETRAIN_PHASE1_DATASET_ROOT}/ja_e-gov_0000_text_document"),
    (8, 772429478, f"{PRETRAIN_PHASE1_DATASET_ROOT}/ja_kaken_0000_text_document"),
    (8, 673493046, f"{PRETRAIN_PHASE1_DATASET_ROOT}/ja_kokkai-giji_0000_text_document"),
    (8, 16255530591, f"{PRETRAIN_PHASE1_DATASET_ROOT}/ja_nwc2010_0000_text_document"),
    (8, 25862823840, f"{PRETRAIN_PHASE1_DATASET_ROOT}/ja_nwjc_0000_text_document"),
    (8, 60813844215, f"{PRETRAIN_PHASE1_DATASET_ROOT}/ja_patent_0000_text_document"),
    (8, 11370270531, f"{PRETRAIN_PHASE1_DATASET_ROOT}/ja_sip-comprehensive-html_0000_text_document"),
    (8, 28352330642, f"{PRETRAIN_PHASE1_DATASET_ROOT}/ja_sip-comprehensive-pdf-pdf2text_0000_text_document"),
    (8, 741256291, f"{PRETRAIN_PHASE1_DATASET_ROOT}/ja_warp-html_0000_text_document"),
    (8, 9563719005, f"{PRETRAIN_PHASE1_DATASET_ROOT}/ja_warp-pdf-e0_0000_text_document"),
    (8, 42891810821, f"{PRETRAIN_PHASE1_DATASET_ROOT}/ja_warp-pdf-e0.2_0000_text_document"),
    (8, 1085125338, f"{PRETRAIN_PHASE1_DATASET_ROOT}/ja_wiki_0000_text_document"),

    # Japanese CC/FineWeb
    (4, 49729722349, f"{PRETRAIN_PHASE1_DATASET_ROOT}/ja_cc_0000_text_document"),
    (4, 49369321010, f"{PRETRAIN_PHASE1_DATASET_ROOT}/ja_cc_0001_text_document"),
    (4, 49657420425, f"{PRETRAIN_PHASE1_DATASET_ROOT}/ja_cc_0002_text_document"),
    (4, 50328833323, f"{PRETRAIN_PHASE1_DATASET_ROOT}/ja_cc_0003_text_document"),
    (4, 18329054681, f"{PRETRAIN_PHASE1_DATASET_ROOT}/ja_cc_0004_text_document"),
    (4, 42179433505, f"{PRETRAIN_PHASE1_DATASET_ROOT}/ja_fineweb-2_0000_text_document"),
    (4, 42736865509, f"{PRETRAIN_PHASE1_DATASET_ROOT}/ja_fineweb-2_0001_text_document"),
    (4, 42466190036, f"{PRETRAIN_PHASE1_DATASET_ROOT}/ja_fineweb-2_0002_text_document"),
    (4, 42415830701, f"{PRETRAIN_PHASE1_DATASET_ROOT}/ja_fineweb-2_0003_text_document"),
    (4, 42040441473, f"{PRETRAIN_PHASE1_DATASET_ROOT}/ja_fineweb-2_0004_text_document"),
    (4, 4255815583, f"{PRETRAIN_PHASE1_DATASET_ROOT}/ja_fineweb-2_0005_text_document"),

    # Korean curated datasets
    (8, 352074304, f"{PRETRAIN_PHASE1_DATASET_ROOT}/ko_wiki_0000_text_document"),

    # Korean FineWeb
    (1, 48038910925, f"{PRETRAIN_PHASE1_DATASET_ROOT}/ko_fineweb2_0000_text_document"),

    # Chinese curated datasets
    (8, 740754914, f"{PRETRAIN_PHASE1_DATASET_ROOT}/zh_wiki_0000_text_document"),

    # Chinese FineWeb
    (1, 136502282670, f"{PRETRAIN_PHASE1_DATASET_ROOT}/zh_fineweb2_0000_text_document"),
    (1, 135056311908, f"{PRETRAIN_PHASE1_DATASET_ROOT}/zh_fineweb2_0001_text_document"),
    (1, 138369517441, f"{PRETRAIN_PHASE1_DATASET_ROOT}/zh_fineweb2_0002_text_document"),
    (1, 145115884006, f"{PRETRAIN_PHASE1_DATASET_ROOT}/zh_fineweb2_0003_text_document"),
    (1, 11414468604, f"{PRETRAIN_PHASE1_DATASET_ROOT}/zh_fineweb2_0004_text_document"),
]

PRETRAIN_PHASE2_DATASET = [
    # (0, 375310698, f"{PRETRAIN_PHASE2_DATASET_ROOT}/Laboro-ParaCorpus/Laboro-ParaCorpus_text_document"),
    # (0, 56329875282, f"{PRETRAIN_PHASE2_DATASET_ROOT}/finepdfs/jpn_Jpan_text_document"),

    (4, 111736807942, f"{PRETRAIN_PHASE2_DATASET_ROOT}/stack_v2/train_0_text_document"),
    (4, 113102982494, f"{PRETRAIN_PHASE2_DATASET_ROOT}/stack_v2/train_1_text_document"),
    (4, 112517889644, f"{PRETRAIN_PHASE2_DATASET_ROOT}/stack_v2/train_2_text_document"),
    (4, 110957273163, f"{PRETRAIN_PHASE2_DATASET_ROOT}/stack_v2/train_3_text_document"),
    (4, 114393462648, f"{PRETRAIN_PHASE2_DATASET_ROOT}/stack_v2/train_4_text_document"),
    (4, 113271109799, f"{PRETRAIN_PHASE2_DATASET_ROOT}/stack_v2/train_5_text_document"),
    (4, 44393545648, f"{PRETRAIN_PHASE2_DATASET_ROOT}/stack_v2/train_6_text_document"),

    # (0, 79392227551, f"{PRETRAIN_PHASE2_DATASET_ROOT}/MegaMathProMax/megamath_web_pro_max_text_document"),
    # (0, 33739895211, f"{PRETRAIN_PHASE2_DATASET_ROOT}/MegaMathProMaxOSS/en_megamath-web-pro-max-oss_text_document"),

    (8, 67421295784, f"{PRETRAIN_PHASE2_DATASET_ROOT}/MegaMathProMaxOSS_v2/en_megamath-web-pro-max-oss2_text_document"),

    # (0, 7452241136, f"{PRETRAIN_PHASE2_DATASET_ROOT}/dolmino-mix-1124/math/tinyGSM-MIND-all_text_document"),
    # (0, 35111947, f"{PRETRAIN_PHASE2_DATASET_ROOT}/dolmino-mix-1124/math/dolmino_math_synth-all_text_document"),
    # (0, 3171669, f"{PRETRAIN_PHASE2_DATASET_ROOT}/dolmino-mix-1124/math/gsm8k-all_text_document"),
    # (0, 60036516798, f"{PRETRAIN_PHASE2_DATASET_ROOT}/dolmino-mix-1124/pes2o-all_text_document"),
    # (0, 3857521208, f"{PRETRAIN_PHASE2_DATASET_ROOT}/dolmino-mix-1124/wiki/wiki-all_text_document"),
    # (0, 1483419806, f"{PRETRAIN_PHASE2_DATASET_ROOT}/dolmino-mix-1124/stackexchange/stackexchange-all_text_document"),
    # (0, 18670986447, f"{PRETRAIN_PHASE2_DATASET_ROOT}/dolmino-mix-1124/flan-all_text_document"),
    # (0, 593365106, f"{PRETRAIN_PHASE2_DATASET_ROOT}/llama_nemotron_post_training_dataset/science_text_document"),
    # (0, 12726103808, f"{PRETRAIN_PHASE2_DATASET_ROOT}/llama_nemotron_post_training_dataset/code_text_document"),
    # (0, 23690436148, f"{PRETRAIN_PHASE2_DATASET_ROOT}/llama_nemotron_post_training_dataset/math_text_document"),
    # (0, 10176021, f"{PRETRAIN_PHASE2_DATASET_ROOT}/llama_nemotron_post_training_dataset/safety_text_document"),
    # (0, 20328137, f"{PRETRAIN_PHASE2_DATASET_ROOT}/llama_nemotron_post_training_dataset/chat_text_document"),
    # (0, 1045760060, f"{PRETRAIN_PHASE2_DATASET_ROOT}/stack_math_qa/stackmathqafull-1q1a_text_document"),
    # (0, 21920146190, f"{PRETRAIN_PHASE2_DATASET_ROOT}/cosmopedia_v2/cosmopedia_v2_fineweb_text_document"),
    # (0, 7514097826, f"{PRETRAIN_PHASE2_DATASET_ROOT}/cosmopedia/cosmopedia_web_samples_v2_text_document"),
    # (0, 9270276396, f"{PRETRAIN_PHASE2_DATASET_ROOT}/cosmopedia/cosmopedia_web_samples_v1_text_document"),
    # (0, 69773566226, f"{PRETRAIN_PHASE2_DATASET_ROOT}/llm-jp-IPT/llm-jp-IPT_v0.2_all_text_document"),
    # (0, 69734846334, f"{PRETRAIN_PHASE2_DATASET_ROOT}/llm-jp-IPT/llm-jp-IPT_v0.2_all_wo_jaster_text_document"),
]

PRETRAIN_DATASET = PRETRAIN_PHASE1_DATASET + PRETRAIN_PHASE2_DATASET

MIDTRAIN_PHASE1_DATASET = [
    # llm-jp-midtraining-corpus-v2: llm-jp-IPT-v0.3.2
    (8, 35111947, f"{MIDTRAIN_PHASE1_DATASET_ROOT}/dolmino-mix-1124/math/dolmino_math_synth-all_text_document"),
    (8, 18670986447, f"{MIDTRAIN_PHASE1_DATASET_ROOT}/dolmino-mix-1124/flan-all_text_document"),
    (8, 21664737137, f"{MIDTRAIN_PHASE1_DATASET_ROOT}/llm-jp-IPT_v0.3.2/en_coding_text_document"),
    (8, 944910869, f"{MIDTRAIN_PHASE1_DATASET_ROOT}/llm-jp-IPT_v0.3.2/en_general_text_document"),
    (8, 14821420581, f"{MIDTRAIN_PHASE1_DATASET_ROOT}/llm-jp-IPT_v0.3.2/en_math_text_document"),
    (8, 14972663145, f"{MIDTRAIN_PHASE1_DATASET_ROOT}/llm-jp-IPT_v0.3.2/en_reasoning_text_document"),
    (8, 13190609080, f"{MIDTRAIN_PHASE1_DATASET_ROOT}/llm-jp-IPT_v0.3.2/ja_coding_text_document"),
    (8, 7214271422, f"{MIDTRAIN_PHASE1_DATASET_ROOT}/llm-jp-IPT_v0.3.2/ja_general_text_document"),
    (8, 7200671993, f"{MIDTRAIN_PHASE1_DATASET_ROOT}/llm-jp-IPT_v0.3.2/ja_math_text_document"),
    (8, 7027895378, f"{MIDTRAIN_PHASE1_DATASET_ROOT}/llm-jp-IPT_v0.3.2/ja_reasoning_text_document"),

    # llm-jp-midtraining-corpus-v2: llm-jp-IPT-v0.3.2 (duplicated from the base subset)
    (8, 1483419806, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_dolmino-stackexchange_0000_text_document"),
    (8, 3141777, f"{PRETRAIN_PHASE1_DATASET_ROOT}/en_gsm8k_0000_text_document"),

    # llm-jp-midtraining-corpus-v2: ablation targets
    (4, 21920146190, f"{MIDTRAIN_PHASE1_DATASET_ROOT}/cosmopedia_v2/cosmopedia_v2_fineweb_text_document"),
    (1, 90855876577, f"{MIDTRAIN_PHASE1_DATASET_ROOT}/nemotron_post_training_dataset_v1/stem_text_document"),
    (2, 63103930594, f"{MIDTRAIN_PHASE1_DATASET_ROOT}/swallow_code_v2/swallow_code_v2_text_document"),
    (1, 93634003883, f"{MIDTRAIN_PHASE1_DATASET_ROOT}/nemotron_pretraining_code_v2/Synthetic-Code-Review_text_document"),
    (0.5, 273588361940, f"{MIDTRAIN_PHASE1_DATASET_ROOT}/nemotron_pretraining_code_v2/Synthetic-Question-Answering_text_document"),
    (1, 94508025916, f"{MIDTRAIN_PHASE1_DATASET_ROOT}/nemotron_pretraining_code_v2/Synthetic-Rewriting_text_document"),
    (4, 29526139694, f"{MIDTRAIN_PHASE1_DATASET_ROOT}/nemotron_pretraining_code_v2/Synthetic-Student-Teacher_text_document"),
    # unused according to ablation results
    # (0, 35282589580, f"{MIDTRAIN_PHASE1_DATASET_ROOT}/nemotron_pretraining_code_v2/Synthetic-Transpilation_text_document"),

    # llm-jp-midtraining-corpus-v2: other
    (4, 594193896, f"{MIDTRAIN_PHASE1_DATASET_ROOT}/llama_nemotron_post_training_dataset/science_text_document"),
    (4, 6047985669, f"{MIDTRAIN_PHASE1_DATASET_ROOT}/nemotron_post_training_dataset_v2/stem_text_document"),
    (4, 177886465, f"{MIDTRAIN_PHASE1_DATASET_ROOT}/nemotron_post_training_dataset_v2/multilingual_ja_text_document"),
    (4, 1664009729, f"{MIDTRAIN_PHASE1_DATASET_ROOT}/nemotron_instruction_following_chat_v1/nemotron_instruction_following_chat_v1_text_document"),
    (4, 4455181271, f"{MIDTRAIN_PHASE1_DATASET_ROOT}/nemotron_math_v2/nemotron_math_v2_stackflow_text_document"),
    (4, 1045760060, f"{MIDTRAIN_PHASE1_DATASET_ROOT}/stack_math_qa/stackmathqafull-1q1a_text_document"),
    (4, 4451626568, f"{MIDTRAIN_PHASE1_DATASET_ROOT}/dolma3_dolmino_pool/wiki_to_rcqa_text_document"),
    (4, 8134075345, f"{MIDTRAIN_PHASE1_DATASET_ROOT}/nemotron_pretraining_specialized_v1/Nemotron-Pretraining-Wiki-Rewrite_text_document"),
    (4, 7452241136, f"{MIDTRAIN_PHASE1_DATASET_ROOT}/dolmino-mix-1124/math/tinyGSM-MIND-all_text_document"),

    # leaked
    #(4, 43368754, f"{MIDTRAIN_PHASE1_DATASET_ROOT}/0179_llmjp_math/gptoss_120b_20260126_text_document"),
    # safe
    (4, 35908480, f"{MIDTRAIN_PHASE1_DATASET_ROOT}/0179_llmjp_math/gptoss_120b_20260202_text_document"),
]

MIDTRAIN_DATASET = MIDTRAIN_PHASE1_DATASET


def calculate_total_tokens(dataset: list[tuple[float, float, str]]) -> float:
    return sum(int(r * nt) for r, nt, _ in dataset)


PRETRAIN_TOKENS = calculate_total_tokens(PRETRAIN_DATASET)
MIDTRAIN_TOKENS = calculate_total_tokens(MIDTRAIN_DATASET)

logging.info(f"Pretrain tokens: {PRETRAIN_TOKENS:20,d}")
logging.info(f"Midtrain tokens: {MIDTRAIN_TOKENS:20,d}")


def calculate_subset_ratio(dataset: list[tuple[float, float, str]], weight: float) -> list[tuple[int, str]]:
    total_tokens = calculate_total_tokens(dataset)
    return [(weight * r * nt / total_tokens, path) for r, nt, path in dataset if r > 0]


PRETRAIN_SUBSETS = calculate_subset_ratio(PRETRAIN_DATASET, PRETRAIN_DATASET_RATIO)
MIDTRAIN_SUBSETS = calculate_subset_ratio(MIDTRAIN_DATASET, MIDTRAIN_DATASET_RATIO)

logging.info("Pretrain subsets:")
for subset in PRETRAIN_SUBSETS:
    logging.info(f"  {subset}")
logging.info("Midtrain subsets:")
for subset in MIDTRAIN_SUBSETS:
    logging.info(f"  {subset}")

for subset in PRETRAIN_SUBSETS + MIDTRAIN_SUBSETS:
    print(subset[0])
    print(subset[1])
