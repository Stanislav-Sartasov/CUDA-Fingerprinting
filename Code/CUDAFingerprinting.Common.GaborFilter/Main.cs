//namespace CUDAFingerprinting.Common.GaborFilter
//{
//    class Program
//    {
//        private static void Main(string[] args)
//        {
//            //var bmp = Resources.SampleFinger4;
//            //double[,] imgDoubles = ImageHelper.LoadImage(bmp);
//            //imgDoubles.DoNormalization(100, 100);
//            //int[,] imgInts = imgDoubles.Select2D((x => (int)x));
//            //OrientationField.OrientationField orf = new OrientationField.OrientationField(imgInts, 16);
//            //double[,] orient = orf.GetOrientationMatrix(imgInts.GetLength(0), imgInts.GetLength(1));
//            //var res = ImageEnhancement.Enhance(imgDoubles, orient, (double)1 / 9, 32, 8);
//            //var bmp2 = ImageHelper.SaveArrayToBitmap(res);
//            //bmp2.Save("test.bmp", ImageHelper.GetImageFormatFromExtension("test.bmp"));
//            var gaussian = new Common.Filter(7, 1);
//            gaussian.WriteMatrix();
//        }
//    }
//}
