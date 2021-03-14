CUDA_VISIBLE_DEVICES=1 python main.py --essay_prompt_id_train 5 --essay_prompt_id_test 5 --target_model cent_hds_order --init_lr 0.001 > log/cent_hds_order_change_lr/p5_lr1e-3.log
CUDA_VISIBLE_DEVICES=1 python main.py --essay_prompt_id_train 4 --essay_prompt_id_test 4 --target_model cent_hds_order --init_lr 0.001 > log/cent_hds_order_change_lr/p4_lr1e-3.log
