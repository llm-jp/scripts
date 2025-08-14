# Number of repeats
# ja
repeats_ja_wiki=1
repeats_ja_cc=1
repeats_ja_warp_html=1
repeats_ja_warp_pdf_e0=1
repeats_ja_warp_pdf_e0_2=1
repeats_ja_kaken=1
repeats_ja_nwc2010=1
repeats_ja_nwjc=1
repeats_ja_ceek=1
repeats_ja_fineweb2=1
repeats_ja_sip3_html=1
repeats_ja_sip3_pdf_pdf2text=1
repeats_ja_sip3_pdf_surya=1
repeats_ja_patents=1
repeats_ja_egov=1
repeats_ja_kokkai=1
repeats_ja_aozora=1
# en
repeats_en_wiki=1
repeats_en_dolma_gutenberg=1
repeats_en_dolma_wiki=1
repeats_en_dolma_pes2o=1
repeats_en_dolma_reddit=1
repeats_en_olmo_arxiv=1
repeats_en_olmo_openwebmath=1
repeats_en_olmo_algebraicstack=1
repeats_en_mathpile=1
repeats_en_gsm8k=1
repeats_en_finemath_4plus=1
repeats_en_dolmino_stackexchange=1
repeats_en_fineweb_edu_high=1
repeats_en_fineweb_edu_mid=1
repeats_en_fineweb_edu_low=1
# zh
repeats_zh_wiki=1
repeats_zh_fineweb2=1
# ko
repeats_ko_wiki=1
repeats_ko_fineweb2=1
# code
repeats_code_stack=1
repeats_code_olmo_starcoder=1

# Helper function to multiply number of tokens and repeats
function calc() {
    repeat=$1; shift
    tokens=$1; shift
    path=$1; shift

    # Output corpus info only if the total number of tokens is greater than 0
    python - << EOF
total = ${repeat}*${tokens}
if total > 0:
    print(f"{total} ${path}" if total > 0 else "")
EOF
}

TRAIN_DATA_0111="/groups/gcg51557/experiments/0111_v4-setup/corpus/tokenized"
TRAIN_DATA_0171="/groups/gcg51557/experiments/0171_corpus-v4-stat/tokenized"

TRAIN_DATA_PATH=(
    #
    # ja
    #

    $(calc ${repeats_ja_wiki} 1052435592 ${TRAIN_DATA_0171}/v3.0b1/data/ja_wiki_0000_text_document)

    $(calc ${repeats_ja_cc} 51275078737 ${TRAIN_DATA_0171}/v3.0b1/data/ja_cc_0000_text_document)
    $(calc ${repeats_ja_cc} 51050803189 ${TRAIN_DATA_0171}/v3.0b1/data/ja_cc_0001_text_document)
    $(calc ${repeats_ja_cc} 50966925226 ${TRAIN_DATA_0171}/v3.0b1/data/ja_cc_0002_text_document)
    $(calc ${repeats_ja_cc} 51393330936 ${TRAIN_DATA_0171}/v3.0b1/data/ja_cc_0003_text_document)
    $(calc ${repeats_ja_cc} 18945423909 ${TRAIN_DATA_0171}/v3.0b1/data/ja_cc_0004_text_document)

    $(calc ${repeats_ja_warp_html} 781256214 ${TRAIN_DATA_0171}/v3.0b1/data/ja_warp-html_0000_text_document)

    $(calc ${repeats_ja_warp_pdf_e0} 10101720729 ${TRAIN_DATA_0171}/v3.0b1/data/ja_warp-pdf-e0_0000_text_document)

    $(calc ${repeats_ja_warp_pdf_e0_2} 48034276069 ${TRAIN_DATA_0171}/v3.0b1/data/ja_warp-pdf-e0.2_0000_text_document)

    $(calc ${repeats_ja_kaken} 893991450 ${TRAIN_DATA_0171}/v3.0b1/data/ja_kaken_0000_text_document)

    $(calc ${repeats_ja_nwc2010} 16557525308 ${TRAIN_DATA_0171}/v3.0b1/data/ja_nwc2010_0000_text_document)

    $(calc ${repeats_ja_nwjc} 26583221369 ${TRAIN_DATA_0171}/v3.0b1/data/ja_nwjc_0000_text_document)

    $(calc ${repeats_ja_ceek} 13237986510 ${TRAIN_DATA_0171}/v3.0b1/data/ja_ceek-news_0000_text_document)

    $(calc ${repeats_ja_fineweb2} 46077181511 ${TRAIN_DATA_0171}/v3.0b1/data/ja_fineweb-2_0000_text_document)
    $(calc ${repeats_ja_fineweb2} 46552946090 ${TRAIN_DATA_0171}/v3.0b1/data/ja_fineweb-2_0001_text_document)
    $(calc ${repeats_ja_fineweb2} 46394027367 ${TRAIN_DATA_0171}/v3.0b1/data/ja_fineweb-2_0002_text_document)
    $(calc ${repeats_ja_fineweb2} 46242789711 ${TRAIN_DATA_0171}/v3.0b1/data/ja_fineweb-2_0003_text_document)
    $(calc ${repeats_ja_fineweb2} 45954072229 ${TRAIN_DATA_0171}/v3.0b1/data/ja_fineweb-2_0004_text_document)
    $(calc ${repeats_ja_fineweb2} 4641212226 ${TRAIN_DATA_0171}/v3.0b1/data/ja_fineweb-2_0005_text_document)

    $(calc ${repeats_ja_sip3_html} 11734843767 ${TRAIN_DATA_0171}/v3.0b1/data/ja_sip-comprehensive-html_0000_text_document)

    $(calc ${repeats_ja_sip3_pdf_pdf2text} 30163183003 ${TRAIN_DATA_0171}/v3.0b1/data/ja_sip-comprehensive-pdf-pdf2text_0000_text_document)

    $(calc ${repeats_ja_sip3_pdf_surya} 158072480 ${TRAIN_DATA_0171}/v3.0b1/data/ja_sip-comprehensive-pdf-surya_0000_text_document)

    $(calc ${repeats_ja_patents} 68674608762 ${TRAIN_DATA_0171}/v3.0b1/data/ja_patent_0000_text_document)

    $(calc ${repeats_ja_egov} 83588358 ${TRAIN_DATA_0171}/v3.0b1/data/ja_e-gov_0000_text_document)

    $(calc ${repeats_ja_kokkai} 786325973 ${TRAIN_DATA_0171}/v3.0b1/data/ja_kokkai-giji_0000_text_document)

    $(calc ${repeats_ja_aozora} 133309110 ${TRAIN_DATA_0171}/v3.0b1/data/ja_aozorabunko_0000_text_document)

    #
    # en
    #

    $(calc ${repeats_en_wiki} 4744259830 ${TRAIN_DATA_0111}/en/en_wiki_0000_text_document)

    $(calc ${repeats_en_dolma_gutenberg} 5494262694 ${TRAIN_DATA_0111}/en/en_dolma-books_0000_text_document)

    $(calc ${repeats_en_dolma_wiki} 3896965449 ${TRAIN_DATA_0111}/en/en_dolma-wiki_0000_text_document)

    $(calc ${repeats_en_dolma_pes2o} 62853772802 ${TRAIN_DATA_0111}/en/en_dolma-pes2o_0000_text_document)

    $(calc ${repeats_en_dolma_reddit} 83015186637 ${TRAIN_DATA_0111}/en/en_dolma-reddit_0000_text_document)

    $(calc ${repeats_en_olmo_arxiv} 22219529548 ${TRAIN_DATA_0111}/en/en_olmo-arxiv_0000_text_document)

    $(calc ${repeats_en_olmo_openwebmath} 13395295861 ${TRAIN_DATA_0111}/en/en_olmo-openwebmath_0000_text_document)

    $(calc ${repeats_en_olmo_algebraicstack} 13280211413 ${TRAIN_DATA_0111}/en/en_olmo-algebraicstack_0000_text_document)

    $(calc ${repeats_en_mathpile} 9176535715 ${TRAIN_DATA_0111}/en/en_mathpile_0000_text_document)

    $(calc ${repeats_en_gsm8k} 2781710 ${TRAIN_DATA_0111}/en/en_gsm8k_0000_text_document)

    $(calc ${repeats_en_finemath_4plus} 10335599308 ${TRAIN_DATA_0111}/en/en_finemath-4plus_0000_text_document)

    $(calc ${repeats_en_dolmino_stackexchange} 1464772187 ${TRAIN_DATA_0111}/en/en_dolmino-stackexchange_0000_text_document)

    $(calc ${repeats_en_fineweb_edu_high} 88542160974 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score25_0000_text_document)
    $(calc ${repeats_en_fineweb_edu_high} 87180401690 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score25_0001_text_document)
    $(calc ${repeats_en_fineweb_edu_high} 65322514272 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score25_0002_text_document)
    $(calc ${repeats_en_fineweb_edu_high} 66476480337 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score26_0000_text_document)
    $(calc ${repeats_en_fineweb_edu_high} 65470244892 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score26_0001_text_document)
    $(calc ${repeats_en_fineweb_edu_high} 50424144779 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score26_0002_text_document)
    $(calc ${repeats_en_fineweb_edu_high} 66591004569 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score27_0000_text_document)
    $(calc ${repeats_en_fineweb_edu_high} 65564685416 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score27_0001_text_document)
    $(calc ${repeats_en_fineweb_edu_high} 55949238107 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score27_0002_text_document)
    $(calc ${repeats_en_fineweb_edu_high} 88297085715 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score28_0000_text_document)
    $(calc ${repeats_en_fineweb_edu_high} 53919503304 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score28_0001_text_document)
    $(calc ${repeats_en_fineweb_edu_high} 66402991536 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score29_0000_text_document)
    $(calc ${repeats_en_fineweb_edu_high} 60034208085 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score29_0001_text_document)
    $(calc ${repeats_en_fineweb_edu_high} 66504586684 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score30_0000_text_document)
    $(calc ${repeats_en_fineweb_edu_high} 62786629615 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score30_0001_text_document)
    $(calc ${repeats_en_fineweb_edu_high} 96512355443 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score31_0000_text_document)
    $(calc ${repeats_en_fineweb_edu_high} 97566002683 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score32_0000_text_document)
    $(calc ${repeats_en_fineweb_edu_high} 71711623749 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score33_0000_text_document)
    $(calc ${repeats_en_fineweb_edu_high} 61306678510 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score34_0000_text_document)
    $(calc ${repeats_en_fineweb_edu_high} 59527990360 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score35_0000_text_document)
    $(calc ${repeats_en_fineweb_edu_high} 41424351165 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score36_0000_text_document)
    $(calc ${repeats_en_fineweb_edu_high} 37975308943 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score37_0000_text_document)
    $(calc ${repeats_en_fineweb_edu_high} 24436522649 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score38_0000_text_document)
    $(calc ${repeats_en_fineweb_edu_high} 18014340088 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score39_0000_text_document)
    $(calc ${repeats_en_fineweb_edu_high} 32740215055 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score40_0000_text_document)

    $(calc ${repeats_en_fineweb_edu_mid} 87382858609 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score15_0000_text_document)
    $(calc ${repeats_en_fineweb_edu_mid} 87983162911 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score15_0001_text_document)
    $(calc ${repeats_en_fineweb_edu_mid} 86898110797 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score15_0002_text_document)
    $(calc ${repeats_en_fineweb_edu_mid} 86079894304 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score15_0003_text_document)
    $(calc ${repeats_en_fineweb_edu_mid} 85783949431 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score15_0004_text_document)
    $(calc ${repeats_en_fineweb_edu_mid} 85664742431 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score15_0005_text_document)
    $(calc ${repeats_en_fineweb_edu_mid} 85853159085 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score15_0006_text_document)
    $(calc ${repeats_en_fineweb_edu_mid} 64388973363 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score15_0007_text_document)
    $(calc ${repeats_en_fineweb_edu_mid} 63804088056 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score15_0008_text_document)
    $(calc ${repeats_en_fineweb_edu_mid} 43218270257 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score15_0009_text_document)
    $(calc ${repeats_en_fineweb_edu_mid} 87432098700 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score16_0000_text_document)
    $(calc ${repeats_en_fineweb_edu_mid} 87817228801 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score16_0001_text_document)
    $(calc ${repeats_en_fineweb_edu_mid} 86692378983 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score16_0002_text_document)
    $(calc ${repeats_en_fineweb_edu_mid} 85929459997 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score16_0003_text_document)
    $(calc ${repeats_en_fineweb_edu_mid} 85776407086 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score16_0004_text_document)
    $(calc ${repeats_en_fineweb_edu_mid} 85822228637 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score16_0005_text_document)
    $(calc ${repeats_en_fineweb_edu_mid} 86058900978 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score16_0006_text_document)
    $(calc ${repeats_en_fineweb_edu_mid} 78532766543 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score16_0007_text_document)
    $(calc ${repeats_en_fineweb_edu_mid} 87595064819 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score17_0000_text_document)
    $(calc ${repeats_en_fineweb_edu_mid} 87690066098 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score17_0001_text_document)
    $(calc ${repeats_en_fineweb_edu_mid} 86413013191 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score17_0002_text_document)
    $(calc ${repeats_en_fineweb_edu_mid} 85975459491 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score17_0003_text_document)
    $(calc ${repeats_en_fineweb_edu_mid} 85931240036 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score17_0004_text_document)
    $(calc ${repeats_en_fineweb_edu_mid} 86100462669 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score17_0005_text_document)
    $(calc ${repeats_en_fineweb_edu_mid} 82767873624 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score17_0006_text_document)
    $(calc ${repeats_en_fineweb_edu_mid} 87803725042 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score18_0000_text_document)
    $(calc ${repeats_en_fineweb_edu_mid} 87503831336 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score18_0001_text_document)
    $(calc ${repeats_en_fineweb_edu_mid} 86360923533 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score18_0002_text_document)
    $(calc ${repeats_en_fineweb_edu_mid} 86036188910 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score18_0003_text_document)
    $(calc ${repeats_en_fineweb_edu_mid} 64673696017 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score18_0004_text_document)
    $(calc ${repeats_en_fineweb_edu_mid} 64716417780 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score18_0005_text_document)
    $(calc ${repeats_en_fineweb_edu_mid} 53671680055 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score18_0006_text_document)
    $(calc ${repeats_en_fineweb_edu_mid} 88033679576 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score19_0000_text_document)
    $(calc ${repeats_en_fineweb_edu_mid} 87061194596 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score19_0001_text_document)
    $(calc ${repeats_en_fineweb_edu_mid} 86285461968 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score19_0002_text_document)
    $(calc ${repeats_en_fineweb_edu_mid} 86315116327 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score19_0003_text_document)
    $(calc ${repeats_en_fineweb_edu_mid} 85007506809 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score19_0004_text_document)
    $(calc ${repeats_en_fineweb_edu_mid} 88238813652 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score20_0000_text_document)
    $(calc ${repeats_en_fineweb_edu_mid} 87182610823 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score20_0001_text_document)
    $(calc ${repeats_en_fineweb_edu_mid} 86442782841 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score20_0002_text_document)
    $(calc ${repeats_en_fineweb_edu_mid} 86534614572 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score20_0003_text_document)
    $(calc ${repeats_en_fineweb_edu_mid} 81306305707 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score20_0004_text_document)
    $(calc ${repeats_en_fineweb_edu_mid} 88275288496 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score21_0000_text_document)
    $(calc ${repeats_en_fineweb_edu_mid} 86935875383 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score21_0001_text_document)
    $(calc ${repeats_en_fineweb_edu_mid} 86612610673 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score21_0002_text_document)
    $(calc ${repeats_en_fineweb_edu_mid} 75618471035 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score21_0003_text_document)
    $(calc ${repeats_en_fineweb_edu_mid} 88428790465 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score22_0000_text_document)
    $(calc ${repeats_en_fineweb_edu_mid} 87114735860 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score22_0001_text_document)
    $(calc ${repeats_en_fineweb_edu_mid} 86744643728 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score22_0002_text_document)
    $(calc ${repeats_en_fineweb_edu_mid} 84391017465 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score22_0003_text_document)
    $(calc ${repeats_en_fineweb_edu_mid} 88377959954 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score23_0000_text_document)
    $(calc ${repeats_en_fineweb_edu_mid} 65232772136 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score23_0001_text_document)
    $(calc ${repeats_en_fineweb_edu_mid} 65238721997 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score23_0002_text_document)
    $(calc ${repeats_en_fineweb_edu_mid} 43569792226 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score23_0003_text_document)
    $(calc ${repeats_en_fineweb_edu_mid} 88379769553 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score24_0000_text_document)
    $(calc ${repeats_en_fineweb_edu_mid} 87047032443 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score24_0001_text_document)
    $(calc ${repeats_en_fineweb_edu_mid} 58403676615 ${TRAIN_DATA_0111}/en-fineweb/en-fineweb_score24_0002_text_document)

    $(calc ${repeats_en_fineweb_edu_low} 101183224177 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2013-20_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 103098200749 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2013-48_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 102640270695 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2014-10_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 95020630644 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2014-15_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 109881651481 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2014-23_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 101972123666 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2014-35_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 105060249490 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2014-41_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 94528986165 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2014-42_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 80870231063 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2014-49_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 99421478317 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2014-52_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 90984269000 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2015-06_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 91959582504 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2015-11_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 85484723146 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2015-14_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 99495808518 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2015-18_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 96614344990 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2015-22_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 84931144536 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2015-27_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 89351430195 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2015-32_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 91852803613 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2015-35_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 72426613399 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2015-40_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 90303648219 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2015-48_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 87108105950 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2016-07_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 75742859667 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2016-18_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 78713853414 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2016-22_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 65196054111 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2016-26_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 87492029033 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2016-30_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 86055464086 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2016-36_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 97386234308 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2016-40_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 119534101318 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2016-44_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 114199388212 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2016-50_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 120092153117 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2017-04_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 123918617774 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2017-09_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 145027482629 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2017-13_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 150993949147 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2017-17_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 111973081268 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2017-22_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 126176717308 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2017-26_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 111238629660 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2017-30_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 131104915487 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2017-34_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 112779090543 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2017-39_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 137130659691 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2017-43_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 115766442753 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2017-47_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 114727406894 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2017-51_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 125187755341 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2018-05_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 124597759646 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2018-09_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 120402377671 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2018-13_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 109848450561 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2018-17_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 101215119974 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2018-22_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 117037405996 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2018-26_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 126958346767 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2018-30_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 101979873337 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2018-34_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 106354904636 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2018-39_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 116717493403 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2018-43_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 109569595936 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2018-47_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 125388343038 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2018-51_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 109086412630 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2019-04_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 115076400763 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2019-09_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 101697607741 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2019-13_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 104216498456 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2019-18_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 106103125067 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2019-22_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 101070222131 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2019-26_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 102305200276 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2019-30_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 111393682027 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2019-35_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 96146993104 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2019-39_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 103513104074 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2019-43_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 96418015360 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2019-47_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 87477260770 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2019-51_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 116902055026 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2020-05_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 89967685908 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2020-10_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 109574420348 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2020-16_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 97309210987 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2020-24_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 118010382032 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2020-29_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 91368645134 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2020-34_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 136382849619 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2020-40_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 104232372710 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2020-45_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 103741376039 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2020-50_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 137334474345 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2021-04_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 113616264414 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2021-10_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 135334784113 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2021-17_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 111280122971 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2021-21_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 105319711763 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2021-25_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 144442651639 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2021-31_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 126263133853 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2021-39_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 146325451697 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2021-43_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 102299673598 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2021-49_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 127229211073 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2022-05_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 156798862638 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2022-21_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 138681924149 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2022-27_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 106919988656 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2022-33_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 147505418378 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2022-40_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 152098945535 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2022-49_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 149532700263 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2023-06_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 151845686008 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2023-14_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 158366601187 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2023-23_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 157200294748 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2023-40_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 153380143010 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2023-50_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 100457190296 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2024-10_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 96773865597 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2024-18_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 93507627100 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2024-22_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 85135385061 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2024-26_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 84705277192 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2024-30_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 73196238311 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2024-33_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 86727219593 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2024-38_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 73119619586 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2024-42_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 79959804610 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2024-46_text_document)
    $(calc ${repeats_en_fineweb_edu_low} 84278288389 ${TRAIN_DATA_0111}/en/en_fineweb-rest_CC-MAIN-2024-51_text_document)

    #
    # zh
    #

    $(calc ${repeats_zh_wiki} 840277331 ${TRAIN_DATA_0111}/zh/zh_wiki_0000_text_document)

    $(calc ${repeats_zh_fineweb2} 190309158473 ${TRAIN_DATA_0111}/zh/zh_fineweb2_0000_text_document)
    $(calc ${repeats_zh_fineweb2} 192160217528 ${TRAIN_DATA_0111}/zh/zh_fineweb2_0001_text_document)
    $(calc ${repeats_zh_fineweb2} 191629318921 ${TRAIN_DATA_0111}/zh/zh_fineweb2_0002_text_document)
    $(calc ${repeats_zh_fineweb2} 198652395168 ${TRAIN_DATA_0111}/zh/zh_fineweb2_0003_text_document)
    $(calc ${repeats_zh_fineweb2} 15248244538 ${TRAIN_DATA_0111}/zh/zh_fineweb2_0004_text_document)

    #
    # ko
    #

    $(calc ${repeats_ko_wiki} 316296219 ${TRAIN_DATA_0111}/ko/ko_wiki_0000_text_document)

    $(calc ${repeats_ko_fineweb2} 51780848623 ${TRAIN_DATA_0111}/ko/ko_fineweb2_0000_text_document)

    #
    # code
    #

    $(calc ${repeats_code_stack} 114051163723 ${TRAIN_DATA_0111}/code/code_stack_0000_text_document)

    $(calc ${repeats_code_olmo_starcoder} 104427769064 ${TRAIN_DATA_0111}/code/code_olmo-starcoder_0000_text_document)
)

unset -f calc
