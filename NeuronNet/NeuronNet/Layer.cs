using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuronNet
{
    class Layer
    {
        public List<Perceptron> _layer = new List<Perceptron>();
        public double perceptronB = 1;
        public List<double> sigma = new List<double>();
        public void AddPerceptron(int kolPerceptron)
        {
            for (int i = 0; i < kolPerceptron; ++i)
            {
                Perceptron _perceptron = new Perceptron();
                _layer.Add(_perceptron);
            }
        }
    }
    class Perceptron
    {
        public List<double> sigmoida = new List<double>();
        public List<double> weight = new List<double>();
        public void Sensor(List<double> sensor, List<double> w, double b)
        {
        //    sigmoida.Clear();
            double sum = 0;
            for (int i = 0; i < sensor.Count(); ++i) sum += (w[i] * sensor[i]);
            sum += b;
            double Tetra = Math.Pow(Math.E, -sum);
            sigmoida.Add(1 / (1 + Tetra));
        }

        public void Weight(int w, Perceptron p)
        {
            p.weight = new List<double>();

            for (int i = 0; i < w; ++i)
            {
                Random rnd = new Random();
                p.weight.Add(rnd.Next(10) + 1);
            }
        }
    }
}
