using System.Drawing;

namespace CUDAFingerprinting.ImageProcessing.Binarization
{
    public static class ImageBinarization
    {
        public static Bitmap Binarize(Bitmap src, int threshold)
        {
            Bitmap bmp = new Bitmap(src.Width, src.Height);
            for (int i = 0; i < bmp.Width; i++)
            {
                for (int j = 0; j < bmp.Height; j++)
                {
                    bmp.SetPixel(i, j, src.GetPixel(i, j).B < threshold ? Color.Black : Color.White);
                }
            }
            return bmp;
        }

        public static int[,] Binarize2D(int[,] src, int threshold)
        {
            int srcWidth = src.GetLength(0);
            int srcHeight = src.GetLength(1);
            int[,] imgInt = new int[srcWidth, srcHeight];

            for (int i = 0; i < srcWidth; i++)
            {
                for (int j = 0; j < srcHeight; j++)
                {
                    imgInt[i, j] = src[i, j] < threshold ? 0 : 255;
                }
            }
            return imgInt;
        }

        public static double[,] Binarize2D(double[,] src, int threshold)
        {
            int srcWidth = src.GetLength(0);
            int srcHeight = src.GetLength(1);
            double[,] imgDouble = new double[srcWidth, srcHeight];
            for (int i = 0; i < srcWidth; i++)
            {
                for (int j = 0; j < srcHeight; j++)
                {
                    imgDouble[i, j] = src[i, j] < threshold ? 0 : 255;
                }
            }
            return imgDouble;
        }
    }
}
