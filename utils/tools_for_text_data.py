import datetime
import os

def get_exp_name(args):
    name = [args.dataset_name, '' if args.bert == 'bert-base' else args.bert, f"lr{args.lr}", f"wd{args.weight_decay}",
            args.loss_name]
    if args.dataset_name in ['wiki500k', 'amazon670k']:
        name.append('t'+str(args.group_y_group))

    return '_'.join([i for i in name if i != ''])




class Logger:
    def __init__(self, name):
        self.name = name

    def log(self, text):
        save_path = f'./logs/text_logs/{self.name}'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, 'a') as f:
            f.write(datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S') + text + '\n')
