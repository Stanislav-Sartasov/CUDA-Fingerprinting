using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuronNet
{
    class NeuroNet
    {
        public List<Layer> NeuNet = new List<Layer>();
        public void makeFirstLayer(int kolPerc)
        {
            Layer firstLayer = new Layer();
            for (int i = 0; i < kolPerc; ++i)
            {
                Perceptron firstPerceptron = new Perceptron();
                firstLayer._layer.Add(firstPerceptron);
            }
            NeuNet.Add(firstLayer);
        }

        public void AddLayer(int kolPerc, int prev)
        {
            Layer layer = new Layer();
            layer.AddPerceptron(kolPerc);
            NeuNet.Add(layer);
        }
        public void DelLayer()
        {
            NeuNet.Remove(NeuNet.Last());
        }

        public double KullbackLeibler(NeuroNet neo, int m, Layer InLayer)
        {
            Random rnd = new Random();
            double p = rnd.Next(1) / 100 + 0.1;
            double _p = 0;
            double KL = 0;

            for (int j = 0; j < InLayer._layer.Count(); ++j)
            {
                _p = 0;
                for (int i = 0; i < m; ++i)
                {
                    _p += InLayer._layer[j].sigmoida[i];
                }
                KL += p * Math.Log10(p / _p) + (1 - p) * (Math.Log10((1 - p)) / (1 - _p));
            }

            return KL;
        }

        public double Autoencoder(NeuroNet n, Layer InLayer, Layer OutLayer, int m)
        {
            double betta = 1;
            double Jsae = 0;
            for (int j = 0; j < m; ++j)
            {
                for (int i = 0; i < OutLayer._layer.Count(); ++i)
                {
                    Jsae += Math.Pow(OutLayer._layer[i].sigmoida[j] - InLayer._layer[i].sigmoida[j], 2);
                }
            }
            Jsae = Jsae * ((double)1 / (double)InLayer._layer.Count());
            Jsae += (betta * KullbackLeibler(n, m, InLayer));
            return Jsae;
        }

        public void changeWeight(NeuroNet neo, int k, int m, List<double>[] inP)
        {
            double lambda = 1;
            double alpha = 1;
            for (int ii = 0; ii < neo.NeuNet[k - 1]._layer.Count(); ++ii)
            {
                for (int jj = 0; jj < neo.NeuNet[k]._layer.Count(); ++jj)
                {
                    double sumWK = 0, pBK = 0;

                    for (int x = 0; x < m; ++x)
                    {
                        neo.NeuNet[k].sigma.Clear();
                        neo.NeuNet[k - 1].sigma.Clear();
                        /////////////////// уровень к
                        double sig = 0;
                        for (int i = 0; i < neo.NeuNet[k]._layer.Count(); ++i)
                        {
                            sig = sig - (inP[x][i] - neo.NeuNet[k]._layer[i].sigmoida[x]) * (neo.NeuNet[k]._layer[i].sigmoida[x]) * (1 - neo.NeuNet[k]._layer[i].sigmoida[x]);
                            neo.NeuNet[k].sigma.Add(sig);
                        }
                        //////////////здесь рассматриваем уровень к-1
                        for (int i = 0; i < neo.NeuNet[k - 1]._layer.Count(); ++i)
                        {
                            sig = 0;
                            for (int j = 1; j < neo.NeuNet[k]._layer.Count(); ++j)
                            {
                                sig += (neo.NeuNet[k - 1]._layer[i].weight[j] * neo.NeuNet[k].sigma[j]);
                            }
                            sig *= ((neo.NeuNet[k - 1]._layer[i].sigmoida[x]) * (1 - neo.NeuNet[k - 1]._layer[i].sigmoida[x]));
                            neo.NeuNet[k - 1].sigma.Add(sig);
                        }
                        ///////////////////////меняем веса, связывающие к-1 и к слои и доп. персептрон В к-1 слоя
                        sumWK += (neo.NeuNet[k - 1]._layer[ii].sigmoida[x] * neo.NeuNet[k].sigma[jj]);
                        pBK += neo.NeuNet[k].sigma[jj];
                    }
                    sumWK /= m;
                    pBK /= m;
                    sumWK = sumWK + lambda * neo.NeuNet[k - 1]._layer[ii].weight[jj];
                    neo.NeuNet[k - 1]._layer[ii].weight[jj] -= (alpha * sumWK);
                    neo.NeuNet[k - 1].perceptronB -= (alpha * pBK);
                }
            }
        }
    }
}
