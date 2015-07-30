using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CUDAFingerprinting.Common;

namespace CUDAFingerprinting.RidgeLine
{
    class Program
    {
        static void Main()
        {
            var bmp = Resources.SampleFinger4;
            var image = ImageHelper.LoadImageAsInt(bmp);

            var detectingMinutias = new RidgeLine(image, 5, 3);

            for (int i = 16; i < image.GetLength(1) - 16; i++)
            {
                for (int j = 16; j < image.GetLength(0) -16; j++)
                {
                    detectingMinutias.FindMinutiaLine(i * 1000 + j, 5.0, 50);
                }
                //if (i%10 == 0)
                //{
                //    MakeBmp(detectingMinutias._visited);
                //}
            }

            foreach (var minutia in detectingMinutias.GetMinutiaList())
            {
                Console.WriteLine("x = {0}; y = {1}; Type = {2}; Angle = {3}", minutia.X, minutia.Y, minutia.Type, minutia.Angle);
            }

            MakeBmp(detectingMinutias._visited);
        }

        static void MakeBmp(bool[,] visited)
        {
            int[,] image = new int[visited.GetLength(0),visited.GetLength(1)];

            for (int i = 0; i < image.GetLength(1); i++)
            {
                for (int j = 0; j < image.GetLength(0); j++)
                {
                    if (visited[j, i])
                    {
                        image[j, i] = 255;
                    }
                }
            }

            ImageHelper.SaveArray(image,"visit.bmp");
        }
    }
}
