# VGGface dataset

# for group in age age-race gender gender-age gender-age-race gender-race race
#     do
#         python vggface2-pred.py --image-dir /home/lsf/桌面/UTK-Face/maskfacegan/vggface/${group}/results_orig --save-dir ./VGGFace/vggface/${group}
#     done

# for group in age age-race gender gender-age gender-age-race gender-race race
#     do
#         python eval_vgg_attribute_all.py --test-set VGGFace/vggface/${group}/vgg-attribute-model-label.txt --save-dir Result-VGGFace2/vggface/${group}
#     done

# Celeba dataset
for group in age age-race gender gender-age gender-age-race gender-race race
    do
        python vggface2-pred.py --image-dir /home/lsf/桌面/UTK-Face/maskfacegan/celeba/${group}/results_celeba_orig --save-dir ./VGGFace/celeba/${group}
    done

for group in age age-race gender gender-age gender-age-race gender-race race
    do
        python eval_vgg_attribute_all.py --test-set VGGFace/celeba/${group}/vgg-attribute-model-label.txt --save-dir Result-VGGFace2/celeba/${group}
    done

