import os
import numpy
import torch
from torch.autograd import Variable
import skipthoughts

class Tester():

   @staticmethod
   def launch_all_tests(uniskip):
      print('Test: '+str(uniskip)+'\n')
      Tester.launch_tests(uniskip)

      #print('With lengths processed')
      #Tester.launch_tests(uniskip)

   @staticmethod
   def launch_tests(uniskip):
      methods = [func for func in dir(Tester) if callable(getattr(Tester, func)) and 'test_' == func[:5]]
      for name in methods:
         print(name)
         getattr(Tester, name)(uniskip)
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
   def test_oneWord(uniskip):
      input = Variable(torch.LongTensor(1,1))
      input.data.zero_()
      input.data[0,0] = 1
      output = uniskip(input)
      output_gt = torch.from_numpy(numpy.load(os.path.join(dir_test,'features_oneWord_normFalse_eosFalse.npy')))
      Tester.eq(output.data[0], output_gt[0,:2400])

   @staticmethod
   def test_oneWord_zeroPadding(uniskip):
      input = Variable(torch.LongTensor(1,3))
      input.data.zero_()
      input.data[0,0] = 1
      output = uniskip(input, lengths=[1])
      output_gt = torch.from_numpy(numpy.load(os.path.join(dir_test,'features_oneWord_normFalse_eosFalse.npy')))
      Tester.eq(output.data[0], output_gt[0,:2400])

   @staticmethod
   def test_oneWord_eos(uniskip):
      input = Variable(torch.LongTensor(1,10))
      input.data.zero_()
      input.data[0,0] = 1
      input.data[0,1] = 5
      output = uniskip(input, lengths=[2])
      output_gt = torch.from_numpy(numpy.load(os.path.join(dir_test,'features_oneWord_normFalse_eosTrue.npy')))
      Tester.eq(output.data[0], output_gt[0,:2400])

   @staticmethod
   def test_words(uniskip):
      input = Variable(torch.LongTensor(1,3))
      input.data.zero_()
      input.data[0,0] = 1
      input.data[0,1] = 2
      input.data[0,2] = 4
      output = uniskip(input, lengths=[3])
      output_gt = torch.from_numpy(numpy.load(os.path.join(dir_test,'features_normFalse_eosFalse.npy')))
      Tester.eq(output.data[0], output_gt[0,:2400])

   @staticmethod
   def test_words_nolengths(uniskip):
      input = Variable(torch.LongTensor(1,3))
      input.data.zero_()
      input.data[0,0] = 1
      input.data[0,1] = 2
      input.data[0,2] = 4
      output = uniskip(input)
      output_gt = torch.from_numpy(numpy.load(os.path.join(dir_test,'features_normFalse_eosFalse.npy')))
      Tester.eq(output.data[0], output_gt[0,:2400])

   @staticmethod
   def bgru_test_words_dropout(uniskip):
      input = Variable(torch.LongTensor(1,3))
      input.data.zero_()
      input.data[0,0] = 1
      input.data[0,1] = 2
      input.data[0,2] = 4

      uniskip.rnn.set_dropout(0.75)
      print(str(uniskip))
      output = uniskip(input)
      output_gt = torch.from_numpy(numpy.load(os.path.join(dir_test,'features_normFalse_eosFalse.npy')))
      Tester.neq(output.data[0], output_gt[0,:2400])

      # uniskip.rnn.set_dropout(1)
      # output = uniskip(input)
      # Tester.eq(output.data[0], output.data[0].clone().fill_(0))

      uniskip.rnn.set_dropout(0)

   @staticmethod
   def test_backprop(uniskip):
      uniskip.zero_grad()
      input = Variable(torch.LongTensor(1,3))
      input.data.zero_()
      input.data[0,0] = 1
      input.data[0,1] = 2
      input.data[0,2] = 4
      output = uniskip(input, lengths=[3])
      loss = output.sum()
      loss.backward()
      saved_grad = [params.grad.data.clone() for params in uniskip.parameters()]

      # test if backprop all the way
      for idx, params in enumerate(uniskip.parameters()):
         Tester.neq(params.grad.data.clone().fill_(0), params.grad.data)

      uniskip.zero_grad()
      input_pad = Variable(torch.LongTensor(1,10))
      input_pad.data.zero_()
      input_pad.data[0,0] = 1
      input_pad.data[0,1] = 2
      input_pad.data[0,2] = 4
      output_pad = uniskip(input_pad, lengths=[3])
      loss_pad = output_pad.sum()
      loss_pad.backward()

      # test if backprop works the same way
      # on padded input and non padded input
      for idx, params in enumerate(uniskip.parameters()):
         Tester.eq(saved_grad[idx], params.grad.data)

if __name__ == '__main__':
   dir_st = '/home/cadene/data/skip-thoughts'
   dir_test = '../data/test'
   vocab = ['robots', 'are', 'very', 'cool', '<eos>', 'BiDiBu']

   path_uniskip = os.path.join(dir_st, 'uniskip.pth')
   uniskip = skipthoughts.UniSkip(dir_st, vocab, dropout=0)
   drop_uniskip = skipthoughts.DropUniSkip(dir_st, vocab, dropout=0)
   bayes_uniskip = skipthoughts.BayesianUniSkip(dir_st, vocab, dropout=0)

   if not os.path.exists(dir_test):
      os.system('python2 theano/dump_features.py')

   Tester.launch_all_tests(uniskip)
   Tester.launch_all_tests(drop_uniskip)
   Tester.launch_all_tests(bayes_uniskip)
   Tester.bgru_test_words_dropout(bayes_uniskip)