# Celeba dataset
# for group in age age-race gender gender-age gender-age-race gender-race race
#     do
#         python celeba-pred.py --image-dir /home/lsf/桌面/UTK-Face/maskfacegan/celeba/${group}/results_celeba_orig --save-dir ./Celeb-A/${group}
#     done

for group in age age-race gender gender-age gender-age-race gender-race race
    do
        python eval_celeba_attribute_all.py --test-set Celeb-A/celeba/${group}/celeba-attribute-model-label.txt --save-dir Result-Celeb-A/celeba/${group}
    done


# VGG dataset
for group in age age-race gender gender-age gender-age-race gender-race race
    do
        python celeba-pred.py --image-dir /home/lsf/桌面/UTK-Face/maskfacegan/vggface/${group}/results_orig --save-dir ./Celeb-A/vggface/${group}
    done

for group in age age-race gender gender-age gender-age-race gender-race race
    do
        python eval_celeba_attribute_all.py --test-set Celeb-A/vggface/${group}/celeba-attribute-model-label.txt --save-dir Result-Celeb-A/vggface/${group}
    done