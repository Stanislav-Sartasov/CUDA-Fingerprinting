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
        private Directions _direction;

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

        private void NewSection(Tuple<int, int> point)  //maybe need remake this func...
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


                if (!OutOfImage(xs, ys) && _image[ys, xs] < 15 && !rightE)
                {
                    _section[_lengthWings - i] = new Tuple<int, int>(xs, ys);
                }
                else
                {
                    rightE = true;
                    _sectionsCenter -= i;
                }
                
                if (!OutOfImage(xe, ye) && _image[ys, xs] < 15 && !leftE)
                {
                    _section[_lengthWings + i] = new Tuple<int, int>(xe, ye);
                }
                else
                {
                    leftE = true;
                    _sectionsCenter += i;
                }

                _sectionsCenter /= 2;

                x = _section[_sectionsCenter].Item1;
                y = _section[_sectionsCenter].Item2;

                _sectionsAngle = _orientation.GetOrientation(x, y);
            }
        }

        private bool OutOfImage(int x, int y)
        {
            return x < 0 || y < 0 || x >= _image.GetLength(1) || y >= _image.GetLength(0);
        }

        private Tuple<int, int> MakeStep()
        {
            int x = _section[_sectionsCenter].Item1;
            int y = _section[_sectionsCenter].Item2;

            double angle = _sectionsAngle + ((int)_direction) * Math.PI + (_diffAngle ? Math.PI : 0) + Math.PI * 2;
            while (angle > Math.PI * 2) angle -= Math.PI * 2;

            double dx = x + _step * Math.Cos(angle);
            double dy = y + _step * Math.Sin(angle);

            x = (int)(dx >= 0 ? dx + 0.5 : dx - 0.5);
            y = (int)(dy >= 0 ? dy + 0.5 : dy - 0.5);

            if (!OutOfImage(x, y)) return new Tuple<int, int>(-1, -1);

            double ang2Check = _orientation.GetOrientation(x, y);
            if ((_sectionsAngle * ang2Check < 0) ? (_sectionsAngle > 1.4) || (ang2Check > 1.4) : Math.Abs(_sectionsAngle - ang2Check) > 0.5)
            {
                _diffAngle = !_diffAngle;
            }

            return new Tuple<int, int>(x, y);
        }

        private void Paint(Tuple<int, int>[] edge, int edgesCenter) //in process
        {
            List<Tuple<int, int>> queue = new List<Tuple<int, int>> {edge[edgesCenter]};

            List<Tuple<int, int>> stopPixels = edge.Where(tuple => tuple.Item1 != -1).ToList();
            stopPixels.AddRange(_section.Where(tuple => tuple.Item1 != -1));

            while (queue.Count > 0)
            {
                int x = queue[0].Item1;
                int y = queue[0].Item2;

                queue.RemoveAt(0);

                
            }
        }

        private MinutiaTypes CheckStopCriteria(int threshold = 100) 
        {
            int x = _section[_sectionsCenter].Item1;
            int y = _section[_sectionsCenter].Item2;

            if (_visited[y, x])
            {
                MakeTestBmp(x, y); 
                return MinutiaTypes.Intersection;
            }

            if (_image[y, x] < threshold)
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

            bmp.SetPixel(x, 364 - y, Color.Red); //What the 364??

            bmp.Save("Test1.bmp");
        }

        private void FollowLine(Tuple<int, int> point, Directions direction)
        {
            NewSection(point);
            if (_section[_lengthWings].Item1 == -1) return; //?

            _direction = direction;

            MinutiaTypes minutiaType;
            int x, y;
            double angle;

            do
            {
                x = point.Item1;
                y = point.Item2;
                angle = _orientation.GetOrientation(x, y);

                var oldSection = _section;
                var oldCenter = _sectionsCenter;

                point = MakeStep();

                angle += ((int)direction) * Math.PI + (_diffAngle ? Math.PI : 0);
                while (angle > Math.PI * 2) angle -= Math.PI * 2;

                if (point.Item1 == -1) return;

                NewSection(point);
                if (_section[_lengthWings].Item1 == -1) return; //?

                minutiaType = CheckStopCriteria();
                Paint(oldSection, oldCenter);
            } while (minutiaType == MinutiaTypes.NotMinutia);

            x = point.Item1;
            y = point.Item2;

            AddMinutia(x, y, angle, minutiaType); //Add check minutia
        }

        public List<Tuple<Minutia, MinutiaTypes>> GetMinutiaList()
        {
            return _minutias;
        }

        public void FindMinutia(int x, int y, double duplicateDelta, int colorThreshold = 10)
        {
            if (_image[y, x] >= colorThreshold || _visited[y, x]) return;
            _visited[y, x] = true;

            FollowLine(new Tuple<int, int>(x, y), Directions.Forward);
            FollowLine(new Tuple<int, int>(x, y), Directions.Back);

            _diffAngle = false;
        }

        private void CheckAndDeleteFalseMinutia(Minutia minutia, MinutiaTypes minutiaTypes, double delta = 50)
        {
            var i =
                _minutias.FindIndex(x => Math.Sqrt(Math.Pow(x.Item1.X - minutia.X, 2) + Math.Pow(x.Item1.Y - minutia.Y, 2)) < delta);

            _minutias.RemoveAt(i);
        }

        private bool IsDuplicate(Minutia minutia, MinutiaTypes minutiaTypes, double delta = 50)
        {
            return
                _minutias.Exists(
                    x => Math.Sqrt(Math.Pow(x.Item1.X - minutia.X, 2) + Math.Pow(x.Item1.Y - minutia.Y, 2)) < delta);
        }
    }
}
