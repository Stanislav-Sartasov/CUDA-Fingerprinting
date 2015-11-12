using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuronNet
{
    class Program
    {
        static void changeSigmoida(int k, NeuroNet NEU, int kolPerceptron, List<double> sig, List<double> weig, bool flag)
        {
            if (flag)
            {
                for (int j = 0; j < NEU.NeuNet[k]._layer.Count(); ++j)
                {
                    NEU.NeuNet[k]._layer[j].sigmoida.Clear();
                }
            }
            for (int ii = 0; ii < kolPerceptron; ++ii)
            {
                for (int j = 0; j < NEU.NeuNet[k - 1]._layer.Count(); ++j)
                {
                    weig.Add(NEU.NeuNet[k - 1]._layer[j].weight[ii]);
                }

                for (int xx = 0; xx < NEU.NeuNet[k - 1]._layer[0].sigmoida.Count(); ++xx)
                {
                    for (int j = 0; j < NEU.NeuNet[k- 1]._layer.Count(); ++j)
                    {
                        sig.Add(NEU.NeuNet[k - 1]._layer[j].sigmoida[xx]);
                    }
                    NEU.NeuNet[k]._layer[ii].Sensor(sig, weig, NEU.NeuNet[k - 1].perceptronB);
                    sig.Clear();
                }
                weig.Clear();
            }

        }
        
        static void Main(string[] args)
        {
            NeuroNet NEU = new NeuroNet();
            List<double> weig = new List<double>();
            List<double> sig = new List<double>();
            double x;
            double epsilon = 1;
            int m, n; // number of input vectors, the number of elements in the vector
            m = Int32.Parse(Console.ReadLine());
            n = Int32.Parse(Console.ReadLine());
            List<double>[] input = new List<double>[m];
            int kolLayer = n*2-1; // how many Layer must be?
            int kolPerceptron = n;
            for (int i = 0; i < m; ++i)
            {
                input[i] = new List<double>();
                for (int j = 0; j < n; ++j)
                {
                    x = Double.Parse(Console.ReadLine());
                    input[i].Add(x);
                }
            }
            NEU.makeFirstLayer(kolPerceptron);
            for (int i = 0; i < m; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    NEU.NeuNet[0]._layer[j].sigmoida.Add(input[i][j]);
                }
            }
            --kolPerceptron;

            for (int i = 1; i < kolLayer; ++i)
            {
                NEU.AddLayer(kolPerceptron, i - 1);
                for (int j = 0; j < NEU.NeuNet[i - 1]._layer.Count(); ++j)
                {
                    NEU.NeuNet[i-1]._layer[j].Weight(kolPerceptron, NEU.NeuNet[i-1]._layer[j]);
                }
                changeSigmoida(i, NEU, kolPerceptron, sig, weig, false);
                
                NEU.AddLayer(kolPerceptron + 1, i);
                
                for (int j = 0; j < NEU.NeuNet[i]._layer.Count(); ++j)
                {
                    NEU.NeuNet[i]._layer[j].Weight(kolPerceptron+1, NEU.NeuNet[i]._layer[j]);
                }
                changeSigmoida(i + 1, NEU, kolPerceptron+1, sig, weig, false);
                double e = NEU.Autoencoder(NEU, NEU.NeuNet[i + 1], NEU.NeuNet[i - 1], m);
                
                while (e > epsilon)
                {
                   Console.WriteLine(e);
                    NEU.changeWeight(NEU, i+1, m, input);
                    NEU.changeWeight(NEU, i, m, input);

                    changeSigmoida(i, NEU, kolPerceptron, sig, weig, true);
                    changeSigmoida(i + 1, NEU, kolPerceptron+1, sig, weig, true);
                   
                    e = NEU.Autoencoder(NEU, NEU.NeuNet[i + 1], NEU.NeuNet[i - 1], m);
                }
                Console.WriteLine(e);
                kolPerceptron += (i < kolLayer / 2) ? -1 : 1;
                NEU.DelLayer();

            }
        }
    }
}
