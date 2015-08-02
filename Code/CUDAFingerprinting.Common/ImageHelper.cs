using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;

namespace CUDAFingerprinting.Common
{
    public static class ImageHelper
    {
        public static void MarkMinutiae(Bitmap source, List<Minutia> minutiae, string path)
        {
            var bmp2 = new Bitmap(source.Width, source.Height);
            for (int x = 0; x < bmp2.Width; x++)
            {
                for (int y = 0; y < bmp2.Height; y++)
                {
                    bmp2.SetPixel(x, y, source.GetPixel(x, y));
                }
            }
            var gfx = Graphics.FromImage(bmp2);

            foreach (var pt in minutiae)
            {
                //minutia box
                gfx.DrawRectangle(Pens.Red, pt.X - 1, pt.Y - 1, 2, 2);
            }

            gfx.Save();

            bmp2.Save(path, ImageFormat.Png);   
        }

        public static void MarkMinutiae(string sourcePath, List<Minutia> minutiae, string path)
        {
            MarkMinutiae(new Bitmap(sourcePath), minutiae, path);
        }

        public static void MarkMinutiaeWithDirections(Bitmap source, List<Minutia> minutiae, string path)
        {
            var bmp2 = new Bitmap(source.Width, source.Height);
            for (int x = 0; x < source.Width; x++)
            {
                for (int y = 0; y < source.Height; y++)
                {
                    bmp2.SetPixel(x, y, source.GetPixel(x, y));
                }
            }
            var gfx = Graphics.FromImage(bmp2);

            foreach (var pt in minutiae)
            {
                //minutia direction
                gfx.DrawLine(Pens.Blue, 
                    pt.X, 
                    pt.Y, 
                    Convert.ToInt32(pt.X + Math.Cos(pt.Angle) * 5),
                    Convert.ToInt32(pt.Y - Math.Sin(pt.Angle) * 5)
                );
            }
            //for marking above direction lines
            foreach (var pt in minutiae)
            {
                //minutia point
                gfx.FillRectangle(Brushes.Red, pt.X, pt.Y, 1, 1);
            }

            gfx.Save();

            bmp2.Save(path, ImageFormat.Png);
        }

        public static void MarkMinutiaeWithDirections(string sourcePath, List<Minutia> minutiae, string path)
        {
            MarkMinutiaeWithDirections(new Bitmap(sourcePath), minutiae, path);
        }

        // IMPORTANT NOTE: The image is stored with (0,0) being top left angle
        // For the simplicity of geometric transformations everywhere in the project
        // the origin point is BOTTOM left angle.
        public static double[,] LoadImage(Bitmap bmp)
        {
            double[,] imgBytes = new double[bmp.Height, bmp.Width];
            for (int x = 0; x < bmp.Width; x++)
            {
                for (int y = 0; y < bmp.Height; y++)
                {
                    imgBytes[bmp.Height-1 - y, x] = bmp.GetPixel(x, y).R;
                }
            }
            return imgBytes;
        }

        public static float[,] LoadImageToFloats(Bitmap bmp)
        {
            float[,] imgBytes = new float[bmp.Height, bmp.Width];
            for (int x = 0; x < bmp.Width; x++)
            {
                for (int y = 0; y < bmp.Height; y++)
                {
                    imgBytes[bmp.Height - 1 - y, x] = bmp.GetPixel(x, y).R;
                }
            }
            return imgBytes;
        }

        // IMPORTANT NOTE: The image is stored with (0,0) being top left angle
        // For the simplicity of geometric transformations everywhere in the project
        // the origin point is BOTTOM left angle.
        public static int[,] LoadImageAsInt(Bitmap bmp)
        {
            int[,] imgBytes = new int[bmp.Height, bmp.Width];
            for (int x = 0; x < bmp.Width; x++)
            {
                for (int y = 0; y < bmp.Height; y++)
                {
                    // the flipping of the image
                    imgBytes[bmp.Height-1 - y, x] = bmp.GetPixel(x, y).R;
                }
            }
            return imgBytes;
        }

        // IMPORTANT NOTE: The image is stored with (0,0) being top left angle
        // For the simplicity of geometric transformations everywhere in the project
        // the origin point is BOTTOM left angle.
        public static double[,] LoadImage(string path)
        {
            return LoadImage(new Bitmap(path));
        }

        public static float[,] LoadImageToFloats(string path)
        {
            return LoadImageToFloats(new Bitmap(path));
        }

        // IMPORTANT NOTE: The image is stored with (0,0) being top left angle
        // For the simplicity of geometric transformations everywhere in the project
        // the origin point is BOTTOM left angle.
        public static int[,] LoadImageAsInt(string path)
        {
            return LoadImageAsInt(new Bitmap(path));
        }

        public static void SaveArray(int[,] data, string path, bool applyNormalization = false)
        {
            SaveArrayToBitmap(data, applyNormalization).Save(path, GetImageFormatFromExtension(path));
        }

        public static void SaveArray(float[,] data, string path, bool applyNormalization = false)
        {
            SaveArrayToBitmap(data, applyNormalization).Save(path, GetImageFormatFromExtension(path));
        }

        public static Bitmap SaveArrayToBitmap(int[,] data, bool applyNormalization = false)
        {
            int x = data.GetLength(1);
            int y = data.GetLength(0);
            var bmp = new Bitmap(x, y);
            data.Select2D((value, row, column) =>
            {
                value = Math.Abs(value);
                // todo: make a proper normalization
                if (applyNormalization)
                    value = (value < 0) ? 0 : (value > 255 ? 255 : value);
                // note: notice the flipping
                lock(bmp)bmp.SetPixel(column, y-1-row, Color.FromArgb(value, value, value));
                return value;
            });
            return bmp;
        }

        public static Bitmap SaveArrayToBitmap(double[,] data, bool applyNormalization = false)
        {
            int x = data.GetLength(1);
            int y = data.GetLength(0);
            var max = data.Max2D();
            var min = data.Min2D();

            var bmp = new Bitmap(x, y);
            data.Select2D((value, row, column) =>
            {
                var gray = applyNormalization?(int)((value - min) / (max - min) * 255):(int)value;
                lock(bmp)
                    bmp.SetPixel(column, bmp.Height - 1 - row, Color.FromArgb(gray, gray, gray));
                return value;
            });
            return bmp;  
        }

        public static Bitmap SaveArrayToBitmap(float[,] data, bool applyNormalization = false)
        {
            int x = data.GetLength(1);
            int y = data.GetLength(0);
            var max = data.Max2D();
            var min = data.Min2D();

            var bmp = new Bitmap(x, y);
            data.Select2D((value, row, column) =>
            {
                var gray = applyNormalization ? (int)((value - min) / (max - min) * 255) : (int)value;
                lock (bmp)
                    bmp.SetPixel(column, bmp.Height - 1 - row, Color.FromArgb(gray, gray, gray));
                return value;
            });
            return bmp;
        }

        public static void SaveArrayAndOpen(double[,] data, string path, bool applyNormalization = false)
        {
            SaveArray(data, path, applyNormalization);
            Process.Start(path);
        }

        public static void SaveArray(double[,] data, string path, bool applyNormalization = false)
        {
            SaveArrayToBitmap(data, applyNormalization).Save(path, GetImageFormatFromExtension(path));
        }

        public static ImageFormat GetImageFormatFromExtension(string path)
        {
            var extension = Path.GetExtension(path).ToUpper();
            switch(extension)
            {
                case ".BMP": return ImageFormat.Bmp;
                case ".JPG":
                case ".JPEG": return ImageFormat.Jpeg;
                case ".PNG": return ImageFormat.Png;
                default: throw new NotSupportedException();
            }
        }
    }
}
