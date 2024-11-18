import os
import sys
import time
from tensorboardX import SummaryWriter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data import create_dataloader
from src.engine.trainer import Trainer
from src.engine.validator import Validator
# from src.options.train_options import TrainOptions
# from src.options.val_options import ValOptions
# from src.options.data_options import TrainDataOptions, ValDataOptions
from src.options import config_settings
from src.engine.strategy.earlystop import EarlyStopping


# Entrance
if __name__ == '__main__':

    # define training and validation options
    # train_opt = TrainOptions().parse()
    # val_opt = ValOptions().parse()
    # train_data_opt = TrainDataOptions().parse()
    # val_data_opt = ValDataOptions().parse()

    opt = config_settings()
    # get data
    data_loader = create_dataloader(opt.train_data)
    val_loader = create_dataloader(opt.val_data)

    # define training settings: optim, loss, model, learning rate, etc.
    trainer = Trainer(opt.train)
    validator = Validator(opt.val)
    validator.update(trainer.model, val_loader)
    
    # record the training summary
    train_writer = SummaryWriter(os.path.join(opt.general.log_dir, opt.general.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.val.log_dir, opt.val.name, "val"))
    train_data_writer = SummaryWriter(os.path.join(opt.general.log_dir, opt.general.name, "train_data"))
    val_data_writer = SummaryWriter(os.path.join(opt.general.log_dir, opt.general.name, "val_data"))

    # set early stopping strategy   
    early_stopping = EarlyStopping(patience=opt.train.earlystop_epoch, delta=-0.001, verbose=True)
    start_time = time.time()
    print ("Length of data loader: %d" %(len(data_loader)))

    # start to training
    for epoch in range(opt.train.epochs):
        
        for i, data in enumerate(data_loader):
            trainer.total_steps += 1

            trainer.set_input(data)
            trainer.optimize_parameters()

            if trainer.total_steps % opt.train.show_loss_freq == 0:
                print("Train loss: {} at step: {}".format(trainer.loss, trainer.total_steps))
                train_writer.add_scalar('loss', trainer.loss, trainer.total_steps)
                print("Iter time: ", ((time.time()-start_time)/trainer.total_steps)  )

            if trainer.total_steps in [10,30,50,100,1000,5000,10000] and False: # save models at these iters 
                model.save_networks('model_iters_%s.pth' % model.total_steps)

        if epoch % opt.train.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d' % (epoch))
            trainer.save_networks( 'model_epoch_best.pth' )
            trainer.save_networks( 'model_epoch_%s.pth' % epoch )

        # Validation
        trainer.eval()
        validator.forward()
        ap, r_acc, f_acc, acc = validator.evaluate()
        val_writer.add_scalar('accuracy', acc, trainer.total_steps)
        val_writer.add_scalar('ap', ap, trainer.total_steps)
        print("(Val @ epoch {}) acc: {}; ap: {}".format(epoch, acc, ap))

        early_stopping(acc, trainer)
        if early_stopping.early_stop:
            cont_train = trainer.adjust_learning_rate()
            if cont_train:
                print("Learning rate dropped by 10, continue training...")
                early_stopping = EarlyStopping(patience=trainer.earlystop_epoch, delta=-0.002, verbose=True)
            else:
                print("Early stopping.")
                break

        trainer.train()