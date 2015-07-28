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
            var bmp = Resources.SampleFinger3;
            var image = ImageHelper.LoadImageAsInt(bmp);

            var detectingMinutias = new RidgeLine(image, 4, 4);

            for (int i = 0; i < image.GetLength(1); i++)
            {
                for (int j = 0; j < image.GetLength(0); j++)
                {
                    detectingMinutias.FindMinutiaLine(i * 1000 + j, 5.0, 125);
                }
            }

            foreach (var minutia in detectingMinutias.GetMinutiaList())
            {
                Console.WriteLine("x = {0}; y = {1}; Type = {2}; Angle = {3}", minutia.X, minutia.Y, minutia.Type, minutia.Angle);
            }
        }
    }
}
