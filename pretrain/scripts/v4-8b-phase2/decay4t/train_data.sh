# v4 pretraining dataset with v4 tokenization

# Helper function to multiply number of tokens and repeats
function calc() {
    repeat=$1; shift
    tokens=$1; shift
    path=$1; shift
    python - << EOF
total = ${repeat} * ${tokens}
if total > 0:
    print(f"{total} ${path}" if total > 0 else "")
EOF
}

DATASET_ROOT="/groups/gcg51557/experiments/0212_v4-train-data/data/v20250816/tokenized"
PHASE2_DATASET_ROOT="/groups/gcg51557/experiments/0193_llmjpv4_midtraining/datasets/tokenized/v4.0_alpha1.0/data"

export TRAIN_DATA_PATH=(
    # Code datasets
    $(calc 8 106882005818 ${DATASET_ROOT}/code_olmo-starcoder_0000_text_document)

    # NOTE(odashi): Stack v1 is replaced with Stack v2 in Phase2.
    #$(calc 8 117852374347 ${DATASET_ROOT}/code_stack_0000_text_document)

    # English curated datasets
    $(calc 8 5139896520 ${DATASET_ROOT}/en_dolma-books_0000_text_document)
    $(calc 8 60036516798 ${DATASET_ROOT}/en_dolma-pes2o_0000_text_document)
    $(calc 8 82254295911 ${DATASET_ROOT}/en_dolma-reddit_0000_text_document)
    $(calc 8 3857521208 ${DATASET_ROOT}/en_dolma-wiki_0000_text_document)
    $(calc 8 1483419806 ${DATASET_ROOT}/en_dolmino-stackexchange_0000_text_document)
    $(calc 8 3141777 ${DATASET_ROOT}/en_gsm8k_0000_text_document)
    $(calc 8 8750035604 ${DATASET_ROOT}/en_mathpile_0000_text_document)
    $(calc 8 12977175126 ${DATASET_ROOT}/en_olmo-algebraicstack_0000_text_document)
    $(calc 8 21716303067 ${DATASET_ROOT}/en_olmo-arxiv_0000_text_document)
    $(calc 8 13171054142 ${DATASET_ROOT}/en_olmo-openwebmath_0000_text_document)
    $(calc 8 4746637139 ${DATASET_ROOT}/en_wiki_0000_text_document)

    # English fineWeb low-scored
    $(calc 1 102568520111 ${DATASET_ROOT}/en_fineweb-rescored_score_10_0000_text_document)
    $(calc 1 102509087783 ${DATASET_ROOT}/en_fineweb-rescored_score_10_0001_text_document)
    $(calc 1 100816401574 ${DATASET_ROOT}/en_fineweb-rescored_score_10_0002_text_document)
    $(calc 1 100065810915 ${DATASET_ROOT}/en_fineweb-rescored_score_10_0003_text_document)
    $(calc 1 99955083033 ${DATASET_ROOT}/en_fineweb-rescored_score_10_0004_text_document)
    $(calc 1 100268985585 ${DATASET_ROOT}/en_fineweb-rescored_score_10_0005_text_document)
    $(calc 1 98582941635 ${DATASET_ROOT}/en_fineweb-rescored_score_10_0006_text_document)
    $(calc 1 102501814684 ${DATASET_ROOT}/en_fineweb-rescored_score_10_0007_text_document)
    $(calc 1 87146714512 ${DATASET_ROOT}/en_fineweb-rescored_score_10_0008_text_document)
    $(calc 1 102540352684 ${DATASET_ROOT}/en_fineweb-rescored_score_11_0000_text_document)
    $(calc 1 101615714055 ${DATASET_ROOT}/en_fineweb-rescored_score_11_0001_text_document)
    $(calc 1 100223579527 ${DATASET_ROOT}/en_fineweb-rescored_score_11_0002_text_document)
    $(calc 1 99832954483 ${DATASET_ROOT}/en_fineweb-rescored_score_11_0003_text_document)
    $(calc 1 100200285660 ${DATASET_ROOT}/en_fineweb-rescored_score_11_0004_text_document)
    $(calc 1 98939237258 ${DATASET_ROOT}/en_fineweb-rescored_score_11_0005_text_document)
    $(calc 1 102361066324 ${DATASET_ROOT}/en_fineweb-rescored_score_11_0006_text_document)
    $(calc 1 100127948990 ${DATASET_ROOT}/en_fineweb-rescored_score_11_0007_text_document)
    $(calc 1 51572633747 ${DATASET_ROOT}/en_fineweb-rescored_score_11_0008_text_document)
    $(calc 1 102346538767 ${DATASET_ROOT}/en_fineweb-rescored_score_12_0000_text_document)
    $(calc 1 100658650543 ${DATASET_ROOT}/en_fineweb-rescored_score_12_0001_text_document)
    $(calc 1 99879930853 ${DATASET_ROOT}/en_fineweb-rescored_score_12_0002_text_document)
    $(calc 1 100207021051 ${DATASET_ROOT}/en_fineweb-rescored_score_12_0003_text_document)
    $(calc 1 99609557924 ${DATASET_ROOT}/en_fineweb-rescored_score_12_0004_text_document)
    $(calc 1 101835220299 ${DATASET_ROOT}/en_fineweb-rescored_score_12_0005_text_document)
    $(calc 1 99887438413 ${DATASET_ROOT}/en_fineweb-rescored_score_12_0006_text_document)
    $(calc 1 91670119172 ${DATASET_ROOT}/en_fineweb-rescored_score_12_0007_text_document)
    $(calc 1 101899332058 ${DATASET_ROOT}/en_fineweb-rescored_score_13_0000_text_document)
    $(calc 1 100107151548 ${DATASET_ROOT}/en_fineweb-rescored_score_13_0001_text_document)
    $(calc 1 100188263808 ${DATASET_ROOT}/en_fineweb-rescored_score_13_0002_text_document)
    $(calc 1 100208220130 ${DATASET_ROOT}/en_fineweb-rescored_score_13_0003_text_document)
    $(calc 1 101460435434 ${DATASET_ROOT}/en_fineweb-rescored_score_13_0004_text_document)
    $(calc 1 99804308223 ${DATASET_ROOT}/en_fineweb-rescored_score_13_0005_text_document)
    $(calc 1 99868561720 ${DATASET_ROOT}/en_fineweb-rescored_score_13_0006_text_document)
    $(calc 1 28118083173 ${DATASET_ROOT}/en_fineweb-rescored_score_13_0007_text_document)
    $(calc 1 101287815642 ${DATASET_ROOT}/en_fineweb-rescored_score_14_0000_text_document)
    $(calc 1 100193448611 ${DATASET_ROOT}/en_fineweb-rescored_score_14_0001_text_document)
    $(calc 1 100909877098 ${DATASET_ROOT}/en_fineweb-rescored_score_14_0002_text_document)
    $(calc 1 100757143238 ${DATASET_ROOT}/en_fineweb-rescored_score_14_0003_text_document)
    $(calc 1 99723340081 ${DATASET_ROOT}/en_fineweb-rescored_score_14_0004_text_document)
    $(calc 1 99265358219 ${DATASET_ROOT}/en_fineweb-rescored_score_14_0005_text_document)
    $(calc 1 15921206946 ${DATASET_ROOT}/en_fineweb-rescored_score_14_0006_text_document)
    $(calc 1 101005877270 ${DATASET_ROOT}/en_fineweb-rescored_score_15_0000_text_document)
    $(calc 1 100489515421 ${DATASET_ROOT}/en_fineweb-rescored_score_15_0001_text_document)
    $(calc 1 101723685894 ${DATASET_ROOT}/en_fineweb-rescored_score_15_0002_text_document)
    $(calc 1 100045434530 ${DATASET_ROOT}/en_fineweb-rescored_score_15_0003_text_document)
    $(calc 1 99720256993 ${DATASET_ROOT}/en_fineweb-rescored_score_15_0004_text_document)
    $(calc 1 98543462382 ${DATASET_ROOT}/en_fineweb-rescored_score_15_0005_text_document)
    $(calc 1 7338842460 ${DATASET_ROOT}/en_fineweb-rescored_score_15_0006_text_document)
    $(calc 1 100825885054 ${DATASET_ROOT}/en_fineweb-rescored_score_16_0000_text_document)
    $(calc 1 101739112228 ${DATASET_ROOT}/en_fineweb-rescored_score_16_0001_text_document)
    $(calc 1 100416142859 ${DATASET_ROOT}/en_fineweb-rescored_score_16_0002_text_document)
    $(calc 1 99754212843 ${DATASET_ROOT}/en_fineweb-rescored_score_16_0003_text_document)
    $(calc 1 100042039542 ${DATASET_ROOT}/en_fineweb-rescored_score_16_0004_text_document)
    $(calc 1 45275202852 ${DATASET_ROOT}/en_fineweb-rescored_score_16_0005_text_document)
    $(calc 1 101199070911 ${DATASET_ROOT}/en_fineweb-rescored_score_17_0000_text_document)
    $(calc 1 101438063034 ${DATASET_ROOT}/en_fineweb-rescored_score_17_0001_text_document)
    $(calc 1 99938292420 ${DATASET_ROOT}/en_fineweb-rescored_score_17_0002_text_document)
    $(calc 1 99892259073 ${DATASET_ROOT}/en_fineweb-rescored_score_17_0003_text_document)
    $(calc 1 88320720228 ${DATASET_ROOT}/en_fineweb-rescored_score_17_0004_text_document)
    $(calc 1 101682793329 ${DATASET_ROOT}/en_fineweb-rescored_score_18_0000_text_document)
    $(calc 1 100779083985 ${DATASET_ROOT}/en_fineweb-rescored_score_18_0001_text_document)
    $(calc 1 99807036301 ${DATASET_ROOT}/en_fineweb-rescored_score_18_0002_text_document)
    $(calc 1 100032616702 ${DATASET_ROOT}/en_fineweb-rescored_score_18_0003_text_document)
    $(calc 1 34415854865 ${DATASET_ROOT}/en_fineweb-rescored_score_18_0004_text_document)
    $(calc 1 101794562305 ${DATASET_ROOT}/en_fineweb-rescored_score_19_0000_text_document)
    $(calc 1 100159790313 ${DATASET_ROOT}/en_fineweb-rescored_score_19_0001_text_document)
    $(calc 1 100003883373 ${DATASET_ROOT}/en_fineweb-rescored_score_19_0002_text_document)
    $(calc 1 56293984847 ${DATASET_ROOT}/en_fineweb-rescored_score_19_0003_text_document)

    # English fineWeb high-scored
    $(calc 4 101909608393 ${DATASET_ROOT}/en_fineweb-rescored_score_20_0000_text_document)
    $(calc 4 100251912003 ${DATASET_ROOT}/en_fineweb-rescored_score_20_0001_text_document)
    $(calc 4 100024677537 ${DATASET_ROOT}/en_fineweb-rescored_score_20_0002_text_document)
    $(calc 4 77617886249 ${DATASET_ROOT}/en_fineweb-rescored_score_20_0003_text_document)
    $(calc 4 101494962972 ${DATASET_ROOT}/en_fineweb-rescored_score_21_0000_text_document)
    $(calc 4 99989476480 ${DATASET_ROOT}/en_fineweb-rescored_score_21_0001_text_document)
    $(calc 4 72610042095 ${DATASET_ROOT}/en_fineweb-rescored_score_21_0002_text_document)
    $(calc 4 101424615382 ${DATASET_ROOT}/en_fineweb-rescored_score_22_0000_text_document)
    $(calc 4 99977160848 ${DATASET_ROOT}/en_fineweb-rescored_score_22_0001_text_document)
    $(calc 4 76067058446 ${DATASET_ROOT}/en_fineweb-rescored_score_22_0002_text_document)
    $(calc 4 100987625784 ${DATASET_ROOT}/en_fineweb-rescored_score_23_0000_text_document)
    $(calc 4 99272893036 ${DATASET_ROOT}/en_fineweb-rescored_score_23_0001_text_document)
    $(calc 4 4407565031 ${DATASET_ROOT}/en_fineweb-rescored_score_23_0002_text_document)
    $(calc 4 100765663494 ${DATASET_ROOT}/en_fineweb-rescored_score_24_0000_text_document)
    $(calc 4 75847299078 ${DATASET_ROOT}/en_fineweb-rescored_score_24_0001_text_document)
    $(calc 4 100728857165 ${DATASET_ROOT}/en_fineweb-rescored_score_25_0000_text_document)
    $(calc 4 73395547895 ${DATASET_ROOT}/en_fineweb-rescored_score_25_0001_text_document)
    $(calc 4 100460689403 ${DATASET_ROOT}/en_fineweb-rescored_score_26_0000_text_document)
    $(calc 4 23787080600 ${DATASET_ROOT}/en_fineweb-rescored_score_26_0001_text_document)
    $(calc 4 100402654509 ${DATASET_ROOT}/en_fineweb-rescored_score_27_0000_text_document)
    $(calc 4 18288732636 ${DATASET_ROOT}/en_fineweb-rescored_score_27_0001_text_document)
    $(calc 4 81686513887 ${DATASET_ROOT}/en_fineweb-rescored_score_28_0000_text_document)
    $(calc 4 65145918651 ${DATASET_ROOT}/en_fineweb-rescored_score_29_0000_text_document)
    $(calc 4 100298736946 ${DATASET_ROOT}/en_fineweb-rescored_score_30_0000_text_document)
    $(calc 4 67634605260 ${DATASET_ROOT}/en_fineweb-rescored_score_30_0001_text_document)

    # Japanese curated datasets
    $(calc 8 124537838 ${DATASET_ROOT}/ja_aozorabunko_0000_text_document)
    $(calc 8 12476129929 ${DATASET_ROOT}/ja_ceek-news_0000_text_document)
    $(calc 8 67690089 ${DATASET_ROOT}/ja_e-gov_0000_text_document)
    $(calc 8 772429478 ${DATASET_ROOT}/ja_kaken_0000_text_document)
    $(calc 8 673493046 ${DATASET_ROOT}/ja_kokkai-giji_0000_text_document)
    $(calc 8 16255530591 ${DATASET_ROOT}/ja_nwc2010_0000_text_document)
    $(calc 8 25862823840 ${DATASET_ROOT}/ja_nwjc_0000_text_document)
    $(calc 8 60813844215 ${DATASET_ROOT}/ja_patent_0000_text_document)
    $(calc 8 11370270531 ${DATASET_ROOT}/ja_sip-comprehensive-html_0000_text_document)
    $(calc 8 28352330642 ${DATASET_ROOT}/ja_sip-comprehensive-pdf-pdf2text_0000_text_document)
    $(calc 8 741256291 ${DATASET_ROOT}/ja_warp-html_0000_text_document)
    $(calc 8 9563719005 ${DATASET_ROOT}/ja_warp-pdf-e0_0000_text_document)
    $(calc 8 42891810821 ${DATASET_ROOT}/ja_warp-pdf-e0.2_0000_text_document)
    $(calc 8 1085125338 ${DATASET_ROOT}/ja_wiki_0000_text_document)

    # Japanese CC/FineWeb
    $(calc 4 49729722349 ${DATASET_ROOT}/ja_cc_0000_text_document)
    $(calc 4 49369321010 ${DATASET_ROOT}/ja_cc_0001_text_document)
    $(calc 4 49657420425 ${DATASET_ROOT}/ja_cc_0002_text_document)
    $(calc 4 50328833323 ${DATASET_ROOT}/ja_cc_0003_text_document)
    $(calc 4 18329054681 ${DATASET_ROOT}/ja_cc_0004_text_document)
    $(calc 4 42179433505 ${DATASET_ROOT}/ja_fineweb-2_0000_text_document)
    $(calc 4 42736865509 ${DATASET_ROOT}/ja_fineweb-2_0001_text_document)
    $(calc 4 42466190036 ${DATASET_ROOT}/ja_fineweb-2_0002_text_document)
    $(calc 4 42415830701 ${DATASET_ROOT}/ja_fineweb-2_0003_text_document)
    $(calc 4 42040441473 ${DATASET_ROOT}/ja_fineweb-2_0004_text_document)
    $(calc 4 4255815583 ${DATASET_ROOT}/ja_fineweb-2_0005_text_document)

    # Korean curated datasets
    $(calc 8 352074304 ${DATASET_ROOT}/ko_wiki_0000_text_document)

    # Korean FineWeb
    $(calc 1 48038910925 ${DATASET_ROOT}/ko_fineweb2_0000_text_document)

    # Chinese curated datasets
    $(calc 8 740754914 ${DATASET_ROOT}/zh_wiki_0000_text_document)

    # Chinese FineWeb
    $(calc 1 136502282670 ${DATASET_ROOT}/zh_fineweb2_0000_text_document)
    $(calc 1 135056311908 ${DATASET_ROOT}/zh_fineweb2_0001_text_document)
    $(calc 1 138369517441 ${DATASET_ROOT}/zh_fineweb2_0002_text_document)
    $(calc 1 145115884006 ${DATASET_ROOT}/zh_fineweb2_0003_text_document)
    $(calc 1 11414468604 ${DATASET_ROOT}/zh_fineweb2_0004_text_document)

    # Phase2 datasets
    #$(calc 0 375310698 ${PHASE2_DATASET_ROOT}/Laboro-ParaCorpus/Laboro-ParaCorpus_text_document)
    #$(calc 0 56329875282 ${PHASE2_DATASET_ROOT}/finepdfs/jpn_Jpan_text_document)

    $(calc 4 111736807942 ${PHASE2_DATASET_ROOT}/stack_v2/train_0_text_document)
    $(calc 4 113102982494 ${PHASE2_DATASET_ROOT}/stack_v2/train_1_text_document)
    $(calc 4 112517889644 ${PHASE2_DATASET_ROOT}/stack_v2/train_2_text_document)
    $(calc 4 110957273163 ${PHASE2_DATASET_ROOT}/stack_v2/train_3_text_document)
    $(calc 4 114393462648 ${PHASE2_DATASET_ROOT}/stack_v2/train_4_text_document)
    $(calc 4 113271109799 ${PHASE2_DATASET_ROOT}/stack_v2/train_5_text_document)
    $(calc 4 44393545648 ${PHASE2_DATASET_ROOT}/stack_v2/train_6_text_document)
    
    #$(calc 0 79392227551 ${PHASE2_DATASET_ROOT}/MegaMathProMax/megamath_web_pro_max_text_document)
    #$(calc 0 33739895211 ${PHASE2_DATASET_ROOT}/MegaMathProMaxOSS/en_megamath-web-pro-max-oss_text_document)

    $(calc 8 67421295784 ${PHASE2_DATASET_ROOT}/MegaMathProMaxOSS_v2/en_megamath-web-pro-max-oss2_text_document)

    #$(calc 0 7452241136 ${PHASE2_DATASET_ROOT}/dolmino-mix-1124/math/tinyGSM-MIND-all_text_document)
    #$(calc 0 35111947 ${PHASE2_DATASET_ROOT}/dolmino-mix-1124/math/dolmino_math_synth-all_text_document)
    #$(calc 0 3171669 ${PHASE2_DATASET_ROOT}/dolmino-mix-1124/math/gsm8k-all_text_document)
    #$(calc 0 60036516798 ${PHASE2_DATASET_ROOT}/dolmino-mix-1124/pes2o-all_text_document)
    #$(calc 0 3857521208 ${PHASE2_DATASET_ROOT}/dolmino-mix-1124/wiki/wiki-all_text_document)
    #$(calc 0 1483419806 ${PHASE2_DATASET_ROOT}/dolmino-mix-1124/stackexchange/stackexchange-all_text_document)
    #$(calc 0 18670986447 ${PHASE2_DATASET_ROOT}/dolmino-mix-1124/flan-all_text_document)
    #$(calc 0 593365106 ${PHASE2_DATASET_ROOT}/llama_nemotron_post_training_dataset/science_text_document)
    #$(calc 0 12726103808 ${PHASE2_DATASET_ROOT}/llama_nemotron_post_training_dataset/code_text_document)
    #$(calc 0 23690436148 ${PHASE2_DATASET_ROOT}/llama_nemotron_post_training_dataset/math_text_document)
    #$(calc 0 10176021 ${PHASE2_DATASET_ROOT}/llama_nemotron_post_training_dataset/safety_text_document)
    #$(calc 0 20328137 ${PHASE2_DATASET_ROOT}/llama_nemotron_post_training_dataset/chat_text_document)
    #$(calc 0 1045760060 ${PHASE2_DATASET_ROOT}/stack_math_qa/stackmathqafull-1q1a_text_document)
    #$(calc 0 21920146190 ${PHASE2_DATASET_ROOT}/cosmopedia_v2/cosmopedia_v2_fineweb_text_document)
    #$(calc 0 7514097826 ${PHASE2_DATASET_ROOT}/cosmopedia/cosmopedia_web_samples_v2_text_document)
    #$(calc 0 9270276396 ${PHASE2_DATASET_ROOT}/cosmopedia/cosmopedia_web_samples_v1_text_document)
    #$(calc 0 69773566226 ${PHASE2_DATASET_ROOT}/llm-jp-IPT/llm-jp-IPT_v0.2_all_text_document)
    #$(calc 0 69734846334 ${PHASE2_DATASET_ROOT}/llm-jp-IPT/llm-jp-IPT_v0.2_all_wo_jaster_text_document)
)
