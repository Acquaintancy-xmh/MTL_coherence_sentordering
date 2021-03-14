CUDA_VISIBLE_DEVICES=2 python main.py --essay_prompt_id_train 6 --essay_prompt_id_test 6 --target_model cent_hds_order --init_lr 0.001 > log/cent_hds_order_change_lr/p6_lr1e-3.log
CUDA_VISIBLE_DEVICES=2 python main.py --essay_prompt_id_train 3 --essay_prompt_id_test 3 --target_model cent_hds_order --init_lr 0.001 > log/cent_hds_order_change_lr/p3_lr1e-3.log
