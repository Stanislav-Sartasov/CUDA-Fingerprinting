using CUDAFingerprinting.Common;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CUDAFingerprinting.RidgeLine
{
    enum Directions {Forward, Back}
    enum MinutiaTypes {NotMinutia, LineEnding, Intersection}

    class RidgeOnLine
    {
        private const double Pi4 = Math.PI/4;

        private List<Tuple<Minutia, MinutiaTypes>> _minutias;
        private int[,] _image;
        private PixelwiseOrientationField _orientation;
        private bool[,] _visited;
        private int _step;
        private int _lengthWings;
        private bool _diffAngle;

        private Tuple<int, int>[] _section;
        private double _sectionsAngle; //?
        private int _sectionsCenter;

        private void AddMinutia(int x, int y, double angle, MinutiaTypes type)
        {
            Minutia newMinutia = new Minutia
            {
                X = x,
                Y = y,
                Angle = angle
            };

            _minutias.Add(new Tuple<Minutia, MinutiaTypes>(newMinutia, type));
        }

        public RidgeOnLine(int[,] image, int step, int lengthWings)
        {
            _image = image;
            _orientation = new PixelwiseOrientationField(image, 16); //will try change blockSize
            _step = step;
            _lengthWings = lengthWings;
            _section = new Tuple<int, int>[2 * lengthWings + 1];
            _diffAngle = false;
            _visited = new bool[image.GetLength(0), image.GetLength(1)];
            _minutias = new List<Tuple<Minutia, MinutiaTypes>>();
        }

        private void NewSection(Tuple<int, int> point)
        {
            _sectionsCenter = 2 * _lengthWings; //At the end func this var will be division in 2 for save section's center

            int x = point.Item1;
            int y = point.Item2;

            bool rightE = false;
            bool leftE = false;

            double angle = _orientation.GetOrientation(x, y) + Pi4*2;

            _section[_lengthWings] = point;

            for (var i = 1; i <= _lengthWings; i++)
            {
                int xs = Convert.ToInt32(x - i * Math.Cos(angle));
                int ys = Convert.ToInt32(y - i * Math.Sin(angle));
                int xe = Convert.ToInt32(x + i * Math.Cos(angle));
                int ye = Convert.ToInt32(y + i * Math.Sin(angle));

                _section[_lengthWings + i] = new Tuple<int, int>(-1, -1);
                _section[_lengthWings - i] = new Tuple<int, int>(-1, -1);


                if (!OutOfImage(xs, ys) && _image[xs, ys] < 15 && !rightE)
                {
                    _section[_lengthWings - i] = new Tuple<int, int>(xs, ys);
                }
                else
                {
                    rightE = true;
                    _sectionsCenter -= i;
                }
                
                if (!OutOfImage(xe, ye) && _image[xs, ys] < 15 && !leftE)
                {
                    _section[_lengthWings + i] = new Tuple<int, int>(xe, ye);
                }
                else
                {
                    leftE = true;
                    _sectionsCenter += i;
                }

                _sectionsCenter /= 2;
            }
        }

        private bool OutOfImage(int x, int y)
        {
            return x < 0 || y < 0 || x >= _image.GetLength(1) || y >= _image.GetLength(0);
        }

        private Tuple<int, int> MakeStep(Tuple<int, int> startPoint, double ang, Directions direction)
        {
            int x = startPoint.Item1;
            int y = startPoint.Item2;

            double angle = ang + ((int)direction) * Math.PI + (_diffAngle ? Math.PI : 0) + Math.PI * 2;
            while (angle > Math.PI * 2) angle -= Math.PI * 2;

            double dx = x + _step * Math.Cos(angle);
            double dy = y + _step * Math.Sin(angle);

            x = (int)(dx >= 0 ? dx + 0.5 : dx - 0.5);
            y = (int)(dy >= 0 ? dy + 0.5 : dy - 0.5);

            if (!OutOfImage(x, y)) return new Tuple<int, int>(-1, -1);

            double ang2Check = _orientation.GetOrientation(x, y);
            if ((ang * ang2Check < 0) ? (ang > 1.4) || (ang2Check > 1.4) : Math.Abs(ang - ang2Check) > 0.5)
            {
                _diffAngle = !_diffAngle;
            }

            return new Tuple<int, int>(x, y);
        }

        private void Paint(Tuple<int, int>[] edge, int edgesCenter)
        {
            
        }

        private MinutiaTypes CheckStopCriteria(int threshold = 100) 
        {
            int x = _section[_sectionsCenter].Item1;
            int y = _section[_sectionsCenter].Item2;

            if (_visited[x, y])
            {
                MakeTestBmp(x, y); 
                return MinutiaTypes.Intersection;
            }

            if (_image[x, y] < threshold)
            {
                MakeTestBmp(x, y);
                return MinutiaTypes.LineEnding;
            }

            return MinutiaTypes.NotMinutia;
        }

        private void MakeTestBmp(int x, int y)  //for visualisation results
        {
            int[,] image = new int[_visited.GetLength(0), _visited.GetLength(1)];

            for (int i = 0; i < image.GetLength(1); i++)
            {
                for (int j = 0; j < image.GetLength(0); j++)
                {
                    if (_visited[j, i])
                    {
                        image[j, i] = 255;
                    }
                }
            }

            var bmp = ImageHelper.SaveArrayToBitmap(image);

            bmp.SetPixel(x, 364 - y, Color.Red); //What??

            bmp.Save("Test1.bmp");
        }

        void
    }
}
