import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import importlib, faulthandler
import inspect,time
from accelerate.utils import DistributedDataParallelKwargs
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

faulthandler.enable()
import utils
from seq_scriptsT import seq_eval
from utils.misc import *
from accelerate import Accelerator


def save_file(arg):
    if not os.path.exists(arg.work_dir) and is_main_process():
        os.makedirs(arg.work_dir)

def judgemettic(arg, dev_wer, best_dev):
    if arg.metric=='wer':
        return dev_wer[arg.metric]<best_dev[arg.metric]
    else:
        return dev_wer[arg.metric]>best_dev[arg.metric]

class Processor():
    def __init__(self, arg):
        self.arg = arg
        save_file(self.arg)
        torch.backends.cudnn.benchmark = True
        if type(self.arg.device) is not int:
            init_distributed_mode(self.arg)
        self.recoder = utils.Recorder(self.arg.work_dir, self.arg.print_log, self.arg.log_interval, arg= arg)
        self.save_arg()
        if self.arg.random_fix:
            self.rng = utils.RandomState(seed=self.arg.random_seed)
        self.device = utils.GpuDataParallel()
        self.dataset = {}
        self.data_loader = {}
        self.model, self.optimizer = self.loading()

    def start(self):
        if self.arg.phase == 'train':
            best_dev = {"wer":200.0, "del":100.0,"ins":100.0, "rouge": 0.0, "bleu1":0.0, "bleu2":0.0,"bleu3":0.0,"bleu4":0.0}
            best_tes = {"wer": 200.0, "del": 100.0, "ins": 100.0,"rouge": 0.0, "bleu1":0.0, "bleu2":0.0,"bleu3":0.0,"bleu4":0.0}
            # best_dev_tran={"rouge": 0.0, "bleu1":0.0, "bleu2":0.0,"bleu3":0.0,"bleu4":0.0}
            # best_tes_tran={"rouge": 0.0, "bleu1":0.0, "bleu2":0.0,"bleu3":0.0,"bleu4":0.0}

            best_epoch,total_time,epoch_time = 0,0,0
            if self.accelerator.is_main_process:
                self.recoder.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            seq_model_list = []

            print("hhh" * 100, self.arg.optimizer_args['start_epoch'], self.arg.optimizer_args['num_epoch'])

            for epoch in range(self.arg.optimizer_args['start_epoch'], self.arg.optimizer_args['num_epoch']):
                save_model = epoch % self.arg.save_interval == 0
                eval_model = epoch % self.arg.eval_interval == 0
                epoch_time = time.time()

                seq_train(self.data_loader['train'], self.model, self.optimizer,
                          epoch, self.recoder,self.accelerator)
                dev_wer={}
                dev_wer['wer']=0
                if eval_model:
                    # train_wer = seq_eval(self.arg, self.data_loader['train_eval'], self.model,
                    #                    'dev', epoch, self.arg.work_dir, self.recoder,self.accelerator)
                    dev_wer = seq_eval(self.arg, self.data_loader['dev'], self.model,
                                       'dev', epoch, self.arg.work_dir, self.recoder,self.accelerator)
                    test_wer = seq_eval(self.arg, self.data_loader['test'], self.model,
                                       'test', epoch, self.arg.work_dir, self.recoder,self.accelerator)
                    if self.accelerator.is_main_process:
                        self.recoder.print_log("Dev WER: {:05.2f}% DEV del {:05.2f}% DEV ins {:05.2f}%".format(dev_wer['wer'], dev_wer['del'], dev_wer['ins']))
                        self.recoder.print_log("Test WER: {:05.2f}% Test del {:05.2f}% Test ins {:05.2f}%".format(test_wer['wer'], test_wer['del'],
                                                                                        test_wer['ins']))
                        # self.recoder.wandb_recoder({'test':test_wer, 'dev':dev_wer})
                        if judgemettic(self.arg, dev_wer, best_dev): #if dev_wer["rouge"] > best_dev["rouge"]: #dev_wer["wer"] < best_dev["wer"]: #
                            best_dev,best_tes, best_epoch = dev_wer,test_wer,epoch

                            model_path = "{}_best_model.pt".format(self.arg.work_dir)
                            self.save_model(epoch, model_path)
                            self.recoder.print_log('Save best model')
                        self.recoder.print_log('Best_dev: {:05.2f}, {:05.2f}, {:05.2f}, '
                                               'Best_test: {:05.2f}, {:05.2f}, {:05.2f},'
                                               'Epoch : {}'.format(best_dev["wer"], best_dev["del"], best_dev["ins"],
                                                                   best_tes["wer"],best_tes["del"],best_tes["ins"], best_epoch))
                        self.recoder.print_log('\t Best_dev. R: {:.4f}  b1:{:.4f}, b2:{:.4f} b3:{:.4f} b4:{:.4f}'.format(best_dev['rouge'], best_dev['bleu1'], best_dev['bleu2'], best_dev['bleu3'], best_dev['bleu4']))
                        self.recoder.print_log('\t Best_test R: {:.4f}  b1:{:.4f}, b2:{:.4f} b3:{:.4f} b4:{:.4f}'.format(best_tes['rouge'], best_tes['bleu1'],  best_tes['bleu2'], best_tes['bleu3'], best_tes[ 'bleu4']))

                        if save_model:
                            print("save ", epoch, " checkpoint")
                            model_path = "{}_epoch_new_model.pt".format(self.arg.work_dir)
                            self.save_model(epoch, model_path)
                        epoch_time = time.time() - epoch_time
                        total_time += epoch_time
                        torch.cuda.empty_cache()
                        self.recoder.print_log('Epoch {} costs {} mins {} seconds'.format(epoch, int(epoch_time)//60, int(epoch_time)%60))
                self.recoder.print_log('Training costs {} hours {} mins {} seconds'.format(int(total_time)//60//60, int(total_time)//60%60, int(total_time)%60))
        elif self.arg.phase == 'test':
            if self.arg.load_weights is None and self.arg.load_checkpoints is None:
                print('Please appoint --weights.')
            self.recoder.print_log('Model:   {}.'.format(self.arg.model))
            self.recoder.print_log('Weights: {}.'.format(self.arg.load_weights))

            dev_wer = seq_eval(self.arg, self.data_loader["dev"], self.model,
                               "dev", 6667, self.arg.work_dir, self.recoder, self.accelerator)
            test_wer = seq_eval(self.arg, self.data_loader["test"], self.model,
                                "test", 6667, self.arg.work_dir, self.recoder, self.accelerator)
            self.recoder.print_log('Evaluation Done.\n')

    def save_arg(self):
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def save_model(self, epoch, save_path):
        # self.accelerator.save_state(save_path)
        model = self.model.module if self.accelerator.num_processes>1 else self.model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.optimizer.scheduler.state_dict(),
            'rng_state': self.rng.save_rng_state(),
        }, save_path)

    def loading(self):
        self.device.set_device(self.arg.device)
        print("Loading model")
        model_class = import_class(self.arg.model)
        model = model_class(
            **self.arg.model_args,
            loss_weights=self.arg.loss_weights,
            cfg = self.arg.dataset_info,
            recoder = self.recoder
        )
        shutil.copy2(inspect.getfile(model_class), self.arg.work_dir)
        optimizer = utils.Optimizer(model, self.arg.optimizer_args)

        if self.arg.load_weights:
            self.load_model_weights(model, self.arg.load_weights)
        elif self.arg.load_checkpoints:
            self.load_checkpoint_weights(model, optimizer)

        self.kernel_sizes = model.conv1d.kernel_size
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model.to(self.arg.local_rank))
        print("Loading model finished.")
        self.load_data()
        kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator= Accelerator(kwargs_handlers=[kwargs])
        model, optimizer.optimizer, optimizer.scheduler = self.accelerator.prepare(
            model, optimizer.optimizer, optimizer.scheduler,
        )
        # self.data_loader['train_eval'] =self.accelerator.prepare(self.data_loader['train_eval'])
        self.data_loader['dev'] =self.accelerator.prepare(self.data_loader['dev'])
        self.data_loader['test'] = self.accelerator.prepare(self.data_loader['test'])
        return model, optimizer


    def load_model_weights(self, model, weight_path):
        # self.accelerator.load_model(weight_path)
        print("loading...", weight_path)
        state_dict = torch.load(weight_path, map_location='cpu')
        if len(self.arg.ignore_weights):
            for w in self.arg.ignore_weights:
                if state_dict.pop(w, None) is not None:
                    print('Successfully Remove Weights: {}.'.format(w))
                else:
                    print('Can Not Remove Weights: {}.'.format(w))
        weights = self.modified_weights(state_dict['model_state_dict'], False)
        s_dict = model.state_dict()
        for name in weights:
            if name not in s_dict:
                print(name,"not in model")
                continue
            if s_dict[name].shape == weights[name].shape:
                s_dict[name] = weights[name]
            else:
                print("shape does not match==", name)
        model.load_state_dict(s_dict, strict=True)

    @staticmethod
    def modified_weights(state_dict, modified=False):
        # state_dict = OrderedDict([(k.replace('.module', ''), v) for k, v in state_dict.items()])
        if not modified:
            return state_dict
        modified_dict = dict()
        return modified_dict

    def load_checkpoint_weights(self, model, optimizer):
        self.load_model_weights(model, self.arg.load_checkpoints)
        state_dict = torch.load(self.arg.load_checkpoints, map_location='cpu')
        if len(torch.cuda.get_rng_state_all()) == len(state_dict['rng_state']['cuda']):
            print("Loading random seeds...")
            self.rng.set_rng_state(state_dict['rng_state'])
        if "optimizer_state_dict" in state_dict.keys():
            print("Loading optimizer parameters...")
            optimizer.load_state_dict(state_dict["optimizer_state_dict"])
            optimizer.to(self.arg.local_rank)
        if "scheduler_state_dict" in state_dict.keys():
            print("Loading scheduler parameters...")
            optimizer.scheduler.load_state_dict(state_dict["scheduler_state_dict"])
        self.arg.optimizer_args['start_epoch'] = state_dict["epoch"] + 1
        self.recoder.print_log("Resuming from checkpoint: epoch {self.arg.optimizer_args['start_epoch']}")

    def load_data(self):
        print("Loading Dataprocessing")
        self.feeder = import_class(self.arg.feeder)
        shutil.copy2(inspect.getfile(self.feeder), self.arg.work_dir)
        if self.arg.dataset == 'CSL':
            dataset_list = zip(["train", "dev"], [True, False])
        else:
            dataset_list = zip(["train", "train_eval", "dev", "test"], [True, False, False, False])
            # dataset_list = zip(["test"], [ False])
        for idx, (mode, train_flag) in enumerate(dataset_list):
            arg = self.arg.feeder_args
            arg["prefix"] = self.arg.dataset_info['dataset_root']
            arg["mode"] = mode.split("_")[0]
            arg["transform_mode"] = train_flag
            arg["annotation_file"] = self.arg.dataset_info[mode.split("_")[0]]
            self.dataset[mode] = self.feeder(kernel_size= self.kernel_sizes, dataset=self.arg.dataset, **arg)
            self.data_loader[mode] = self.build_dataloader(self.dataset[mode], mode, train_flag)

    def init_fn(self, worker_id):
        np.random.seed(int(self.arg.random_seed)+worker_id)


    def build_dataloader(self, dataset, mode, train_flag):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size= self.arg.batch_size if mode == "train" else self.arg.test_batch_size,
            shuffle=train_flag,
            drop_last=train_flag,
            num_workers=self.arg.num_worker,  # if train_flag else 0
            collate_fn=self.feeder.collate_fn,
            pin_memory=False,
            worker_init_fn=self.init_fn,
            persistent_workers=True,
        )


def import_class(name):
    components = name.rsplit('.', 1)
    mod = importlib.import_module(components[0])
    mod = getattr(mod, components[1])
    return mod


if __name__ == '__main__':
    sparser = utils.get_parser()
    p = sparser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            try:
                default_arg = yaml.load(f, Loader=yaml.FullLoader)
            except AttributeError:
                default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k), end="  ")
                assert (k in key)
        sparser.set_defaults(**default_arg)
    args = sparser.parse_args()
    with open(f"./configs/{args.dataset}.yaml", 'r') as f:
        args.dataset_info = yaml.load(f, Loader=yaml.FullLoader)
    processor = Processor(args)


    processor.start()
