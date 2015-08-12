using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CUDAFingerprinting.Common;
using System.Drawing;
using CUDAFingerprinting.Common.OrientationField;
using CUDAFingerprinting.FeatureExtraction.Minutiae;
using CUDAFingerprinting.ImageProcessing.Binarization;
using CUDAFingerprinting.ImageProcessing.Postprocessing;
using CUDAFingerprinting.ImageProcessing.ImageThinning;
using CUDAFingerprinting.ImageProcessing.GaborEnhancement;
using CUDAFingerprinting.ImageProcessing.Segmentation;
using CUDAFingerprinting.FeatureExtraction.Tests;

namespace MinutiaRead
{
    class Program
    {
        static void Main()
        {
            
            string path = "D:\\DB2_bmp\\2_6.bmp";
            
            Bitmap map = new Bitmap(path);
                      
            double[,] mas1 = ImageHelper.LoadImage<double>(map);
            mas1.DoNormalization(100, 100);

            int[,] array = mas1.Select2D(x => (int)x);

            OrientationField orf = new OrientationField(array, 16);//magic const!
            double[,] orient = orf.GetOrientationMatrix(array.GetLength(0), array.GetLength(1));

            var freqMatrix = LocalRidgeFrequency.GetFrequencies(mas1, orient);

            var mas = ImageEnhancement.Enhance(mas1, orient, freqMatrix, 32, 8); // MOAR

            map = ImageHelper.SaveArrayToBitmap<double>(mas);

           
           // map.Save("D:\\test3.bmp");
            map = ImageBinarization.Binarize(map, 128);//i have no idea why

            array = ImageHelper.LoadImage<int>(map);
          
            array = KernelHelper.Make2D<int>(HolesAndIslandsResolver.ResolveHolesAndIslands(
                KernelHelper.Make1D<int>(array), 16, 9, array.GetLength(1), array.GetLength(0)), array.GetLength(0), array.GetLength(1));

            mas = array.Select2D(x => (double)x);
            
            mas = Thinner.Thin(mas, mas.GetLength(1), mas.GetLength(0));
            
            array = mas.Select2D(x => (int)x);
            
            map = ImageHelper.SaveArrayToBitmap<int>(array);
            //map.Save("D:\\test.bmp");
            /*
            Segmentator M = new Segmentator(map);
            byte[,] byteMatrix = M.Segmentate();
            Bitmap bmp = M.MakeBitmap(byteMatrix);
            M.SaveSegmentation(bmp, "D:\\test2.bmp");
            */
            
            
            PixelwiseOrientationField img = new PixelwiseOrientationField(array, 16);//moar magic constants
            
            List<Minutia> mins = MinutiaDetector.GetMinutias(array, img);
            List<Minutia> mins1 = new List<Minutia>();
            int border = 20;
            for (int i = 0; i < mins.Count; ++i)
            {
                if (!(mins[i].X < border || mins[i].X > array.GetLength(1) - border ||
                    mins[i].Y < border || mins[i].Y > array.GetLength(0) - border))
                {
                    mins1.Add(mins[i]);
                }
            }
            //ImageHelper.MarkMinutiae(map, mins1, "D:\\test1.bmp");
            System.IO.StreamWriter write = new System.IO.StreamWriter("D:\\testBug.txt");

            write.WriteLine(mins1.Count);
            for (int i = 0; i < mins1.Count; ++i)
            {
                write.WriteLine(mins1[i].X + " "  + mins1[i].Y + " " + mins1[i].Angle);
            }

            write.Close();/*
            System.IO.StreamReader read = new System.IO.StreamReader("D:\\test2.txt");
            string s;
            
            mins1 = new List<Minutia>();
            s = read.ReadLine();
            int n = Int32.Parse(s);
            float[] masZ = new float[3];
            Minutia m = new Minutia();
            Console.WriteLine(n);
            for (int i = 0; i < n; ++i)
            {
                //file.WriteLine(mins1[i].X + " "  + mins1[i].Y + " " + mins1[i].Angle);
                s = read.ReadLine();
                masZ = s.Split(' ').Select(x => float.Parse(x)).ToArray();
                m.X = (int)masZ[0];
                m.Y = (int)masZ[1];
                m.Angle = masZ[2];
                mins1.Add(m);
            }
            ImageHelper.MarkMinutiae(map, mins1, "D:\\test11.bmp");
            read.Close(); 
            
            Console.WriteLine(Resources.minutia1_1);*/
        }
    }
}
