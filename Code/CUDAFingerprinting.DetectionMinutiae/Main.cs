using System;
using System.Collections.Generic;
using System.Drawing;
using CUDAFingerprinting.Common;

namespace CUDAFingerprinting.DetectionMinutiae
{
    class Program
    {
        static void Main()
        {
            var bmp = Resources.SampleFinger3;
            var image = ImageHelper.LoadImage<int>(bmp);

            var detectingMinutias = new RidgeOnLine(image, 2, 5);  //(image, step, size_wings)

            for (int i = 0; i < image.GetLength(1); i++)
            {
                for (int j = 0; j < image.GetLength(0); j++)
                {
                    detectingMinutias.FindMinutia(i, j, 5.0);  
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
                temp.X = minutia.Item1.X;
                temp.Y = image.GetLength(0) - minutia.Item1.Y - 1;
                temp.Angle = minutia.Item1.Angle;

                Console.WriteLine($@"x = {temp.X}; y = {temp.Y}; Type = {minutia.Item2}; Angle = {temp.Angle}");
                if (minutia.Item2 == MinutiaTypes.LineEnding) MinutiaE.Add(temp);
                if (minutia.Item2 == MinutiaTypes.Intersection) MinutiaI.Add(temp);
            }


	        var visitedMap = detectingMinutias.GetVisitedMap();
            var bmpVis = new Bitmap(MakeBmp(visitedMap));
            bmpVis.Save("Vis.bmp");
            ImageHelper.MarkMinutiae("Vis.bmp", MinutiaE, "withMinutiaE.bmp");
			ImageHelper.MarkMinutiae("Vis.bmp", MinutiaI, "withMinutiaI.bmp");
			Console.ReadKey();
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
