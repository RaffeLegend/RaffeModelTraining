import os
import time
from tensorboardX import SummaryWriter

from data import create_dataloader
from src.engine.trainer import Trainer
from src.engine.validator import Validator
from options.train_options import TrainOptions
from options.val_options import ValOptions
from options.data_options import TrainDataOptions, ValDataOptions
from src.engine.strategy.earlystop import EarlyStopping

# Entrance
if __name__ == '__main__':

    # define training and validation options
    train_opt = TrainOptions().initialize()
    val_opt = ValOptions().initialize()
    train_data_opt = TrainDataOptions().initialize()
    val_data_opt = ValDataOptions().initialize()

     # get data
    data_loader = create_dataloader(train_data_opt)
    val_loader = create_dataloader(val_data_opt)

    # define training settings: optim, loss, model, learning rate, etc.
    trainer = Trainer(train_opt)
    validator = Validator(val_opt).update(trainer.model, val_loader)
    
    # record the training summary
    train_writer = SummaryWriter(os.path.join(train_opt.log_dir, train_opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(val_opt.log_dir, val_opt.name, "val"))
    train_data_writer = SummaryWriter(os.path.join(train_data_opt.log_dir, train_data_opt.name, "train_data"))
    val_data_writer = SummaryWriter(os.path.join(val_data_opt.log_dir, val_data_opt.name, "val_data"))

    # set early stopping strategy   
    early_stopping = EarlyStopping(patience=train_opt.earlystop_epoch, delta=-0.001, verbose=True)
    start_time = time.time()
    print ("Length of data loader: %d" %(len(data_loader)))

    # start to training
    for epoch in range(train_opt.epochs):
        
        for i, data in enumerate(data_loader):
            trainer.total_steps += 1

            trainer.set_input(data)
            trainer.optimize_parameters()

            if trainer.total_steps % train_opt.show_loss_freq == 0:
                print("Train loss: {} at step: {}".format(trainer.loss, trainer.total_steps))
                train_writer.add_scalar('loss', trainer.loss, trainer.total_steps)
                print("Iter time: ", ((time.time()-start_time)/trainer.total_steps)  )

            if trainer.total_steps in [10,30,50,100,1000,5000,10000] and False: # save models at these iters 
                model.save_networks('model_iters_%s.pth' % model.total_steps)

        if epoch % train_opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d' % (epoch))
            trainer.save_networks( 'model_epoch_best.pth' )
            trainer.save_networks( 'model_epoch_%s.pth' % epoch )

        # Validation
        trainer.eval()
        ap, r_acc, f_acc, acc = Validator.evaluation(trainer.model, val_loader)
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