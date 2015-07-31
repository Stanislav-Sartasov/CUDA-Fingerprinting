using System;
using System.Collections.Generic;
using System.Drawing;
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

            var detectingMinutias = new RidgeLine(image, 4, 4);  //(image, step, size_wings)

            for (int i = 0; i < image.GetLength(1); i++)
            {
                for (int j = 0; j < image.GetLength(0); j++)
                {
                    detectingMinutias.FindMinutiaLine(i * 1000 + j, 5.0, 50);  
                }
                //if (i%10 == 0)
                //{
                //    MakeBmp(detectingMinutias._visited);
                //}
            }

            List<Common.Minutia> MinutiaE = new List<Common.Minutia>();  //Create List<EndingLine>
            List<Common.Minutia> MinutiaI = new List<Common.Minutia>();  //Create List<Intersection>

            foreach (var minutia in detectingMinutias.GetMinutiaList())
            {
                Common.Minutia temp = new Common.Minutia();
                temp.X = minutia.X;
                temp.Y = 364 - minutia.Y;
                temp.Angle = minutia.Angle;

                Console.WriteLine("x = {0}; y = {1}; Type = {2}; Angle = {3}", minutia.X, minutia.Y, minutia.Type, minutia.Angle);
                if (minutia.Type == MinutiaTypes.LineEnding) MinutiaE.Add(temp);
                if (minutia.Type == MinutiaTypes.Intersection) MinutiaI.Add(temp);
            }

            var bmpVis = new Bitmap(MakeBmp(detectingMinutias._visited));
            bmpVis.Save("Vis.bmp");
            ImageHelper.MarkMinutiae("Vis.bmp", MinutiaE, MinutiaI, "withMinutia.bmp");
        }

        static Bitmap MakeBmp(bool[,] visited, string name = "visit.bmp")
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

            return ImageHelper.SaveArrayToBitmap(image);
        }
    }
}
