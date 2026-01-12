import torch
import numpy as np
import unittest


class DDPMTests(unittest.TestCase):
    def __init__(self, diffusion, device):
        super().__init__()
        self.diffusion = diffusion
        self.device = device

        self.data_1 = torch.load("unittest_data/data_1.pt")
        self.data_2 = torch.load("unittest_data/data_2.pt")
        self.data_3 = torch.load("unittest_data/data_3.pt")
        self.data_4 = torch.load("unittest_data/data_4.pt")

    def test_q_sample(self, data, msg_postfix):
        x_0 = data['x_0'].to(self.device)
        x_t = data['x_t'].to(self.device)
        t = data['t'].to(self.device)
        eps = data['eps'].to(self.device)

        x_t_pred = self.diffusion.q_sample(x_0, t, noise=eps)

        self.assertEqual(x_t_pred.shape, x_t.shape, f"Zły kształt wyjścia z metody q_sample {msg_postfix}")
        self.assertEqual(x_t_pred.dtype, x_t.dtype, f"Zły typ wyjścia z metody q_sample {msg_postfix}")
        self.assertEqual(x_t_pred.device, x_t.device, f"Zły device wyjścia z metody q_sample {msg_postfix}")
        self.assertTrue(torch.allclose(x_t_pred, x_t, atol=1e-4), 
                        f"Złe wartości wyjścia z metody q_sample {msg_postfix}")

    def test_q_sample_for_2d_data(self):
        self.test_q_sample(self.data_1, "dla danych 2D.")

    def test_q_sample_for_image_data(self):
        self.test_q_sample(self.data_2, "dla danych obrazowych.")

    def test_q_posterior(self, data, msg_postfix):
        x_0 = data['x_0'].to(self.device)
        x_t = data['x_t'].to(self.device)
        t = data['t'].to(self.device)
        eps = data['eps'].to(self.device)
        x_prev = data['x_prev'].to(self.device)

        x_prev_pred = self.diffusion.q_posterior(x_t, x_0, t, noise=eps)

        self.assertEqual(x_prev_pred.shape, x_prev.shape, f"Zły kształt wyjścia z metody q_posterior {msg_postfix}")
        self.assertEqual(x_prev_pred.dtype, x_prev.dtype, f"Zły typ wyjścia z metody q_posterior {msg_postfix}")
        self.assertEqual(x_prev_pred.device, x_prev.device, f"Zły device wyjścia z metody q_posterior {msg_postfix}")
        self.assertTrue(torch.allclose(x_prev_pred, x_prev, atol=1e-4), 
                        f"Złe wartości wyjścia z metody q_posterior {msg_postfix}")

    def test_q_posterior_for_2d_data(self):
        self.test_q_posterior(self.data_3, "dla danych 2D.")

    def test_q_posterior_for_image_data(self):
        self.test_q_posterior(self.data_4, "dla danych obrazowych.")

    def run_tests(self):
        self.test_q_sample_for_2d_data()
        self.test_q_sample_for_image_data()
        self.test_q_posterior_for_2d_data()
        self.test_q_posterior_for_image_data()

    def assess_loss(self, loss_values):
        mean = np.mean(loss_values[50:])

        self.assertTrue(mean > 0.2, "Średnia wartości funkcji straty jest podejrzanie niska.")
        self.assertTrue(mean < 0.3, "Średnia wartości funkcji straty jest podejrzanie wysoka.")


class DDIMTests(unittest.TestCase):
    def __init__(self, diffusion, device):
        super().__init__()
        self.diffusion = diffusion
        self.device = device

        self.data_5 = torch.load("unittest_data/data_5.pt")
        self.data_6 = torch.load("unittest_data/data_6.pt")

    def test_q_posterior(self, data, msg_postfix):
        x_0 = data['x_0'].to(self.device)
        x_t = data['x_t'].to(self.device)
        t = data['timesteps'].to(self.device)
        x_prev = data['x_prev'].to(self.device)

        x_prev_pred = self.diffusion.q_posterior(x_t, x_0, t)

        self.assertEqual(x_prev_pred.shape, x_prev.shape, f"Zły kształt wyjścia z metody q_posterior {msg_postfix}")
        self.assertEqual(x_prev_pred.dtype, x_prev.dtype, f"Zły typ wyjścia z metody q_posterior {msg_postfix}")
        self.assertEqual(x_prev_pred.device, x_prev.device, f"Zły device wyjścia z metody q_posterior {msg_postfix}")
        self.assertTrue(torch.allclose(x_prev_pred, x_prev, atol=1e-4), 
                        f"Złe wartości wyjścia z metody q_posterior {msg_postfix}")

    def test_q_posterior_for_2d_data(self):
        self.test_q_posterior(self.data_5, "dla danych 2D.")

    def test_q_posterior_for_image_data(self):
        self.test_q_posterior(self.data_6, "dla danych obrazowych.")

    def run_tests(self):
        self.test_q_posterior_for_2d_data()
        self.test_q_posterior_for_image_data()