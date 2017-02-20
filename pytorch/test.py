import os
import numpy
import torch
from torch.autograd import Variable
import skipthoughts

def is_uniskip(model):
   for base in model.__class__.__bases__:
      if base == skipthoughts.AbstractUniSkip:
         return True
   return False

def is_biskip(model):
   for base in model.__class__.__bases__:
      if base == skipthoughts.AbstractBiSkip:
         return True
   return False   

class Tester():

   @staticmethod
   def launch_all_tests(model):
      print('Test: '+str(model)+'\n')
      Tester.launch_tests(model)

      #print('With lengths processed')
      #Tester.launch_tests(model)

   @staticmethod
   def launch_tests(model):
      methods = [func for func in dir(Tester) if callable(getattr(Tester, func)) and 'test_' == func[:5]]
      for name in methods:
         print(name)
         getattr(Tester, name)(model)
         print()

   @staticmethod
   def eq(tensor1, tensor2):
      dist = torch.dist(tensor1, tensor2)
      if dist < 1e-5:
         msg = 'success'
      else:
         print(dist)
         msg = 'fail'
      print(msg)

   @staticmethod
   def neq(tensor1, tensor2):
      dist = torch.dist(tensor1, tensor2)
      if dist > 1e-5:
         msg = 'success'
      else:
         print(dist)
         msg = 'fail'
      print(msg)

   @staticmethod
   def _test_skipthoughts(model, output, fname):
      output_gt = torch.from_numpy(numpy.load(os.path.join(dir_test, fname)))
      if is_uniskip(model):
         Tester.eq(output.data[0], output_gt[0,:2400])
      elif is_biskip(model):
         Tester.eq(output.data[0], output_gt[0,2400:])
      else:
         print('Unkown model of type: {} {}'.format(model.__class__, model.__classes__.__bases__))

   @staticmethod
   def test_oneWord(model):
      input = Variable(torch.LongTensor(1,1))
      input.data.zero_()
      input.data[0,0] = 1
      output = model(input)
      Tester._test_skipthoughts(model, output, 'features_oneWord_normFalse_eosFalse.npy')

   @staticmethod
   def test_oneWord_zeroPadding(model):
      input = Variable(torch.LongTensor(1,3))
      input.data.zero_()
      input.data[0,0] = 1
      output = model(input, lengths=[1])
      Tester._test_skipthoughts(model, output, 'features_oneWord_normFalse_eosFalse.npy')

   @staticmethod
   def test_oneWord_eos(model):
      input = Variable(torch.LongTensor(1,10))
      input.data.zero_()
      input.data[0,0] = 1
      input.data[0,1] = 5
      output = model(input, lengths=[2])
      Tester._test_skipthoughts(model, output, 'features_oneWord_normFalse_eosTrue.npy')

   @staticmethod
   def test_words(model):
      input = Variable(torch.LongTensor(1,3))
      input.data.zero_()
      input.data[0,0] = 1
      input.data[0,1] = 2
      input.data[0,2] = 4
      output = model(input, lengths=[3])
      Tester._test_skipthoughts(model, output, 'features_normFalse_eosFalse.npy')

   @staticmethod
   def test_words_nolengths(model):
      input = Variable(torch.LongTensor(1,3))
      input.data.zero_()
      input.data[0,0] = 1
      input.data[0,1] = 2
      input.data[0,2] = 4
      output = model(input)
      Tester._test_skipthoughts(model, output, 'features_normFalse_eosFalse.npy')

   @staticmethod
   def bgru_test_words_dropout(model):
      input = Variable(torch.LongTensor(1,3))
      input.data.zero_()
      input.data[0,0] = 1
      input.data[0,1] = 2
      input.data[0,2] = 4

      model.rnn.set_dropout(0.75)
      print(str(model))
      output = model(input)
      Tester._test_skipthoughts(model, output, 'features_normFalse_eosFalse.npy')

      # model.rnn.set_dropout(1)
      # output = model(input)
      # Tester.eq(output.data[0], output.data[0].clone().fill_(0))

      model.rnn.set_dropout(0)

   @staticmethod
   def test_backprop(model):
      model.zero_grad()
      input = Variable(torch.LongTensor(1,3))
      input.data.zero_()
      input.data[0,0] = 1
      input.data[0,1] = 2
      input.data[0,2] = 4
      output = model(input, lengths=[3])
      loss = output.sum()
      loss.backward()
      saved_grad = [params.grad.data.clone() for params in model.parameters()]

      # test if backprop all the way
      for idx, params in enumerate(model.parameters()):
         Tester.neq(params.grad.data.clone().fill_(0), params.grad.data)

      model.zero_grad()
      input_pad = Variable(torch.LongTensor(1,10))
      input_pad.data.zero_()
      input_pad.data[0,0] = 1
      input_pad.data[0,1] = 2
      input_pad.data[0,2] = 4
      output_pad = model(input_pad, lengths=[3])
      loss_pad = output_pad.sum()
      loss_pad.backward()

      # test if backprop works the same way
      # on padded input and non padded input
      for idx, params in enumerate(model.parameters()):
         Tester.eq(saved_grad[idx], params.grad.data)

if __name__ == '__main__':
   dir_st = '/home/cadene/data/skip-thoughts'
   dir_test = '../data/test'
   vocab = ['robots', 'are', 'very', 'cool', '<eos>', 'BiDiBu']

   if not os.path.exists(dir_test):
      os.system('python2 theano/dump_features.py')

   biskip = skipthoughts.BiSkip(dir_st, vocab, dropout=0)
   Tester.launch_all_tests(biskip)

   uniskip = skipthoughts.UniSkip(dir_st, vocab, dropout=0)
   Tester.launch_all_tests(uniskip)
   
   drop_uniskip = skipthoughts.DropUniSkip(dir_st, vocab, dropout=0)
   Tester.launch_all_tests(drop_uniskip)

   bayes_uniskip = skipthoughts.BayesianUniSkip(dir_st, vocab, dropout=0)
   Tester.launch_all_tests(bayes_uniskip)
   Tester.bgru_test_words_dropout(bayes_uniskip)