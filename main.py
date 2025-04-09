import torch
import sys
import os
import copy
from utility.helper import *
from utility.batch_test import *
import multiprocessing
import torch.multiprocessing
import random
from utility.optimize import HMG
from utility.models import *
from utility.parser import parse_args
from utility.load_data import DataHandler


def get_lables(n_items, temp_set, k=0.9999):
    max_item = 0
    item_lenth = []
    for i in temp_set:
        item_lenth.append(len(temp_set[i]))
        if len(temp_set[i]) > max_item:
            max_item = len(temp_set[i])
    item_lenth.sort()

    max_item = item_lenth[int(len(item_lenth) * k) - 1]

    for i in temp_set:
        if len(temp_set[i]) > max_item:
            temp_set[i] = temp_set[i][0:max_item]
        while len(temp_set[i]) < max_item:
            temp_set[i].append(n_items)
    return max_item, temp_set


def get_train_instances(max_item_list, beh_label_list):
    user_train = []
    beh_item_list = [list() for i in range(n_behs)]  #

    for i in beh_label_list[-1].keys():
        user_train.append(i)
        beh_item_list[-1].append(beh_label_list[-1][i])
        for j in range(n_behs - 1):
            if not i in beh_label_list[j].keys():
                beh_item_list[j].append([n_items] * max_item_list[j])
            else:
                beh_item_list[j].append(beh_label_list[j][i])

    user_train = np.array(user_train)
    beh_item_list = [np.array(beh_item) for beh_item in beh_item_list]
    user_train = user_train[:, np.newaxis]
    return user_train, beh_item_list


def get_train_pairs(user_train_batch, beh_item_tgt_batch):
    input_u_list, input_i_list = [], []
    for i in range(len(user_train_batch)):
        pos_items = beh_item_tgt_batch[i][np.where(beh_item_tgt_batch[i] != n_items)]  # ndarray [x,]
        uid = user_train_batch[i][0]
        input_u_list += [uid] * len(pos_items)
        input_i_list += pos_items.tolist()

    return np.array(input_u_list).reshape([-1]), np.array(input_i_list).reshape([-1])


def test_torch(ua_embeddings, ia_embeddings, rela_embedding, users_to_test, config, batch_test_flag=False):
    def get_score_np(ua_embeddings, ia_embeddings, rela_embedding, users, items):
        ug_embeddings = ua_embeddings[users]  # []
        pos_ig_embeddings = ia_embeddings[items]
        dot = np.multiply(pos_ig_embeddings, rela_embedding)  # [I, dim] * [1, dim]-> [I, dim]
        batch_ratings = np.matmul(ug_embeddings, dot.T)  # [U, dim] * [dim, I] -> [U, I]
        return batch_ratings

    result = {'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)), 'precision': np.zeros(len(Ks)), 
              'hit_ratio': np.zeros(len(Ks))}

    test_users = users_to_test
    n_test_users = len(test_users)

    # pool = torch.multiprocessing.Pool(cores)
    # pool = multiprocessing.Pool(cores)

    u_batch_size = BATCH_SIZE
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0
    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]

        item_batch = range(config['n_items'])
        rate_batch = get_score_np(ua_embeddings, ia_embeddings, rela_embedding, user_batch, item_batch)

        #user_batch_rating_uid = zip(rate_batch, user_batch)
        batch_result = [test_one_user(data, data_generator, config) for data in zip(rate_batch, user_batch)]
        count += len(batch_result)
    
        for re in batch_result:
            result['recall'] += re['recall'] / n_test_users
            result['precision'] += re['precision'] / n_test_users
            result['ndcg'] += re['ndcg'] / n_test_users
            result['hit_ratio'] += re['hit_ratio'] / n_test_users
    
    assert count == n_test_users

    #pool.close()
    return result


def preprocess_sim(args, config):
    topk1_user = args.topk1_user
    topk1_item = args.topk1_item

    input_u_sim = torch.tensor(config['user_sim'])
    user_topk_values1 = torch.topk(input_u_sim, min(topk1_user, config['n_users']), dim=1, largest=True)[
        0][..., -1, None]
    user_indices_remove = input_u_sim > user_topk_values1[..., -1, None]

    input_i_sim = torch.tensor(config['item_sim'])
    item_topk_values1 = torch.topk(input_i_sim, min(topk1_item, config['n_items']), dim=1, largest=True)[
        0]
    item_indices_remove = input_i_sim > item_topk_values1[..., -1, None]
    item_indices_token = torch.tensor([False] * item_indices_remove.shape[0], dtype=torch.bool).reshape(-1, 1)
    item_indices_remove = torch.cat([item_indices_remove, item_indices_token], dim=1)

    return user_indices_remove, item_indices_remove


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    import random
    random.seed(seed)  # Python random module.
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)

    os.environ["GIT_PYTHON_REFRESH"] = "quiet"
    #os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    set_seed(2023)
    
    """
    *********************************************************
    Load Data 
    """
    args = parse_args()
    device = configs['device']
    data_generator = DataHandler(dataset=args.dataset, batch_size=args.batch_size)

    USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
    N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test
    BATCH_SIZE = args.batch_size // 2
    Ks = eval(args.Ks)
    
    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items
    config['behs'] = data_generator.behs
    config['trn_mat'] = data_generator.trnMats[-1] 
    config['Ks'] = Ks
    """
    *********************************************************
    Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
    """
    
    
    pre_adj_list = data_generator.get_adj_mat()
    config['pre_adjs'] = pre_adj_list

    print('use the pre adjcency matrix')
    n_users, n_items = data_generator.n_users, data_generator.n_items
    behs = data_generator.behs
    n_behs = data_generator.beh_num

    # user_sim_mat_unified, item_sim_mat_unified = data_generator.get_unified_sim(args.sim_measure)
    
    # config['user_sim'] = user_sim_mat_unified.todense()
    # config['item_sim'] = item_sim_mat_unified.todense()
    # print("config['user_sim']", config['user_sim'].shape)

    config['user_sim'] = np.ones((config['n_users'], config['n_users']))
    config['item_sim'] = np.ones((config['n_items'], config['n_items']))
    user_indices, item_indices = preprocess_sim(args, config)

    trnDicts = copy.deepcopy(data_generator.trnDicts)
    max_item_list = []
    beh_label_list = []
    for i in range(n_behs):
        max_item, beh_label = get_lables(n_items, trnDicts[i])
        max_item_list.append(max_item)
        beh_label_list.append(beh_label)
    
    t0 = time()

    model = KGMBR(max_item_list, data_config=config, args=args).to(device)

    # kg
    Kg_model = data_generator.Kg_model
    contrast_model = data_generator.contrast_model
    kg_optimizer = data_generator.kg_optimizer

    augmentor = Augmentor(data_config=config, args=args)
    recloss = RecLoss(data_config=config, args=args).to(device)
    ssloss2 = SSLoss2(data_config=config, args=args).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    hmg = HMG(model.parameters(), relax_factor=args.meta_r, beta=args.meta_b)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_gamma)

    run_time = 1
    logger = Logger()
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []

    stopping_step = 0
    should_stop = False
    cur_best_pre_0 = 0.

    user_train1, beh_item_list = get_train_instances(max_item_list, beh_label_list)

    nonshared_idx = -1

    for epoch in range(args.epoch):
        model.train()

        transR_loss = model.kg_init_TATEC(Kg_model.kg_dataset,Kg_model,kg_optimizer,index=0)
        TATEC_loss = model.kg_init_transE(Kg_model.kg_dataset,Kg_model,kg_optimizer,index=1)
        print(f"transE_loss: {transR_loss:.3f}--TATEC_loss: {TATEC_loss:.3f}")

        contrast_views = contrast_model.get_ui_kg_view()

        shuffle_indices = np.random.permutation(np.arange(len(user_train1)))
        user_train1 = user_train1[shuffle_indices]
        beh_item_list = [beh_item[shuffle_indices] for beh_item in beh_item_list]

        t1 = time()
        loss, rec_loss, emb_loss, ssl_loss, ssl2_loss = 0., 0., 0., 0., 0.

        n_batch = int(len(user_train1) / args.batch_size) # 84

        iter_time = time()

        for idx in range(n_batch):
            optimizer.zero_grad()

            start_index = idx * args.batch_size
            end_index = min((idx + 1) * args.batch_size, len(user_train1))

            u_batch = user_train1[start_index:end_index]
            beh_batch = [beh_item[start_index:end_index] for beh_item in
                         beh_item_list]  # [[B, max_item1], [B, max_item2], [B, max_item3]]

            u_batch_list, i_batch_list = get_train_pairs(user_train_batch=u_batch,
                                                         beh_item_tgt_batch=beh_batch[-1])  # ndarray[N, ]  ndarray[N, ]

            # load into cuda
            u_batch = torch.from_numpy(u_batch).to(device)
            beh_batch = [torch.from_numpy(beh_item).to(device) for beh_item in beh_batch]
            u_batch_indices = user_indices[u_batch_list].to(device)  # [B, N]

            i_batch_indices = item_indices[i_batch_list].to(device)  # [B, N]
            u_batch_list = torch.from_numpy(u_batch_list).to(device)
            i_batch_list = torch.from_numpy(i_batch_list).to(device)

            model_time = time()
            ua_embeddings, ia_embeddings, io_embeddings, rela_embeddings, \
            attn_user, attn_item = model(device)
            # print('model time: %.1fs' % (time() - model_time))
            # rec_loss_time = time()
            batch_rec_loss, batch_emb_loss = recloss(u_batch, beh_batch, ua_embeddings, ia_embeddings, rela_embeddings)
            batch_ssl_loss = model.BPR_train_contrast(u_batch_list, i_batch_list,ua_embeddings, io_embeddings,Kg_model,contrast_model,contrast_views)

            batch_ssl2_loss_list = []
            for aux_beh in eval(args.aux_beh_idx):
                aux_beh_ssl2_loss = ssloss2(u_batch_list, i_batch_list, ua_embeddings, ia_embeddings, aux_beh,
                                            u_batch_indices, i_batch_indices)
                batch_ssl2_loss_list.append(aux_beh_ssl2_loss)
            batch_ssl2_loss = sum(batch_ssl2_loss_list)
            batch_loss = batch_rec_loss + batch_emb_loss + batch_ssl_loss + 0.5*batch_ssl2_loss

            if nonshared_idx == -1:
                batch_ssl_loss.backward(retain_graph=True)
                for p_idx, p_name in enumerate(model.all_weights):
                    p = model.all_weights[p_name]
                    if p.grad is None or p.grad.equal(torch.zeros_like(p.grad)):
                        nonshared_idx = p_idx
                        break
                model.zero_grad()

            hmg.step([batch_rec_loss, batch_ssl_loss] + batch_ssl2_loss_list, nonshared_idx)
            batch_emb_loss.backward()
            optimizer.step()

            loss += batch_loss.item() / n_batch
            rec_loss += batch_rec_loss.item() / n_batch
            emb_loss += batch_emb_loss.item() / n_batch
            ssl_loss += batch_ssl_loss.item() / n_batch
            ssl2_loss += batch_ssl2_loss.item() / n_batch

        # print('iter time: %.1fs' % (time() - iter_time))
        if args.lr_decay: scheduler.step()
        torch.cuda.empty_cache()

        if np.isnan(loss) == True:
            print('ERROR: loss is nan.')
            sys.exit()

        # print the test evaluation metrics each 5 epochs; pos:neg = 1:10.
        if (epoch + 1) % args.test_epoch != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f + %.5f]' % (
                    epoch, time() - t1, loss, rec_loss, emb_loss, ssl_loss, ssl2_loss)
                print(perf_str)
            continue  # if (epoch+1)%5 true, excute continue, below eval no longer excute, forward to next epoch

        test_idx = epoch // args.test_epoch
        t2 = time()
        model.eval()
        with torch.no_grad():
            ua_embeddings, ia_embeddings, _, rela_embeddings, attn_user, attn_item = model(device)
            users_to_test = list(data_generator.test_set.keys())
            ret = test_torch(ua_embeddings[:, -1, :].detach().cpu().numpy(),
                             ia_embeddings[:, -1, :].detach().cpu().numpy(),
                             rela_embeddings[behs[-1]].detach().cpu().numpy(), users_to_test, config)

        logger.log_eval(test_idx, ret, Ks)

        if not os.path.exists(args.weights_path):
            os.makedirs(args.weights_path)
        file = f"kgmbr-{args.dataset}-{test_idx}.pth"
        weight_file = os.path.join(args.weights_path,file)
        
        if test_idx >= 0:
            torch.save(model.state_dict(), weight_file)
        
        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])
        print('recall', rec_loger)
        print('ndcg', ndcg_loger)
        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]@[%.1fh]:, recall=[%.5f, %.5f], ' \
                       'ndcg=[%.5f, %.5f], precision=[%.5f, %.5f], hit=[%.5f, %.5f]' % \
                       (
                           epoch, t2 - t1, t3 - t2, (time()-t0)/3600, ret['recall'][0],
                           ret['recall'][1],ret['ndcg'][0], ret['ndcg'][1],
                           ret['precision'][0], ret['precision'][1], ret['hit_ratio'][0], ret['hit_ratio'][1]
                           )
            print(perf_str)

        cur_best_pre_0, stopping_step,should_stop, flag = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                              stopping_step, expected_order='acc',
                                                                              flag_step=10)
        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        # if should_stop == True:
        #     break

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1fh]\trecall=[%s], ndcg=[%s]" % \
                 (idx, (time()-t0)/3600, '\t'.join(['%.4f' % r for r in recs[idx]]),
                  '\t'.join(['%.4f' % r for r in ndcgs[idx]]))
    print(final_perf)
