using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using CUDAFingerprinting.Common;
using CUDAFingerprinting.Common.OrientationField;

namespace CUDAFingerprinting.DetectionMinutiae
{
	internal enum Directions
	{
		Forward,
		Back
	}

	internal enum MinutiaTypes
	{
		NotMinutia,
		LineEnding,
		Intersection
	}

	internal class RidgeOnLine
	{
		private const double Pi4 = Math.PI/4;
		private int _countTest; //for testing
		private Directions _direction;
		private readonly int[,] _image;
		private readonly int _lengthWings;

		private readonly List<Tuple<Minutia, MinutiaTypes>> _minutias;
		private readonly PixelwiseOrientationField _orientation;

		private readonly Tuple<int, int>[] _section;
		private double _sectionsAngle; //?
		private int _sectionsCenter;
		private readonly int _step;
		private readonly bool[,] _visited;

		public RidgeOnLine(int[,] image, int step, int lengthWings)
		{
			_countTest = 0;

			_image = image;
			_orientation = new PixelwiseOrientationField(image, 18); //will try change blockSize
			_step = step;
			_lengthWings = lengthWings;
			_section = new Tuple<int, int>[2*lengthWings + 1];
			_visited = new bool[image.GetLength(0), image.GetLength(1)];
			_minutias = new List<Tuple<Minutia, MinutiaTypes>>();
		}

		private void AddMinutia(Minutia newMinutia, MinutiaTypes type)
		{
			_minutias.Add(new Tuple<Minutia, MinutiaTypes>(newMinutia, type));
		}

		public void FindMinutia(int x, int y, double duplicateDelta, int colorThreshold = 15)
		{
			if ((_image[y, x] >= colorThreshold) || _visited[y, x]) return;
			_visited[y, x] = true;


			_sectionsAngle = _orientation.GetOrientation(y, x);
			if (_sectionsAngle < 0) _sectionsAngle += 2 * Math.PI;
			FollowLine(new Tuple<int, int>(x, y), Directions.Forward);


			_sectionsAngle = _orientation.GetOrientation(y, x) + Math.PI;
			//if (_sectionsAngle < 0) _sectionsAngle += 2 * Math.PI;
			FollowLine(new Tuple<int, int>(x, y), Directions.Back);
		}

		//Need add calculate angle and remade selection 
		private void NewSection(Tuple<int, int> point)
		{
			for (var index = 0; index < _section.Length; index++)
				_section[index] = new Tuple<int, int>(-1, -1);

			var x = point.Item1;
			var y = point.Item2;

			var lEnd = _lengthWings;
			var rEnd = lEnd;

			var rightE = false;
			var leftE = false;

			var angle = _orientation.GetOrientation(y, x);
			angle += Pi4;

			_section[_lengthWings] = point; //check on withe??

			for (var i = 1; i <= _lengthWings; i++)
			{
				var xs = Convert.ToInt32(x - i*Math.Cos(angle));
				var ys = Convert.ToInt32(y - i*Math.Sin(angle));
				var xe = Convert.ToInt32(x + i*Math.Cos(angle));
				var ye = Convert.ToInt32(y + i*Math.Sin(angle));

				if (!OutOfImage(xs, ys) && (_image[ys, xs] < 15) && !rightE)
				{
					_section[_lengthWings - i] = new Tuple<int, int>(xs, ys);
					rEnd--;
				}
				else
				{
					rightE = true;
				}

				if (!OutOfImage(xe, ye) && (_image[ye, xe] < 15) && !leftE)
				{
					_section[_lengthWings + i] = new Tuple<int, int>(xe, ye);
					lEnd++;
				}
				else
				{
					leftE = true;
				}

				_sectionsCenter = (lEnd + rEnd)/2;

				x = _section[_sectionsCenter].Item1;
				y = _section[_sectionsCenter].Item2;

				angle = _orientation.GetOrientation(y, x);
				//Console.WriteLine(angle);
				angle += (double)_direction * Math.PI;
				if (angle < 0) angle += 2 * Math.PI;

				if (Math.Abs(_sectionsAngle - angle) > 0.2) angle += Math.PI;
				while (angle > 2 * Math.PI) angle -= Math.PI;

				_sectionsAngle = angle;
				//Console.WriteLine(_sectionsAngle);
			}
		}

		private bool OutOfImage(int x, int y)
		{
			return (x < 0) || (y < 0) || (x >= _image.GetLength(1)) || (y >= _image.GetLength(0));
		}

		//Remade step
		private Tuple<int, int> MakeStep(int _x = -1, int _y = -1)
		{
			var x = _section[_sectionsCenter].Item1;
			var y = _section[_sectionsCenter].Item2;

			var step = _step;

			var angle = _sectionsAngle;

			var dx = x + step * Math.Cos(angle);
			var dy = y + step * Math.Sin(angle);

			x = (int) (dx >= 0 ? dx + 0.5 : dx - 0.5);
			y = (int) (dy >= 0 ? dy + 0.5 : dy - 0.5);

			return OutOfImage(x, y) ? new Tuple<int, int>(-1, -1) : new Tuple<int, int>(x, y);
		}

		private void Paint(Tuple<int, int>[] edge)
		{
			var queue = new List<Tuple<int, int>>();
			Tuple<int, int> v1, v2;

			int x1 = -1,x2 = -1, y1 = -1, y2 = -1, x_a, y_a;

			foreach (var tuple in edge.Where(tuple => tuple.Item1 != -1))
			{
				if (x1 == -1)
				{
					x1 = tuple.Item1;
					y1 = tuple.Item2;
				}

				x2 = tuple.Item1;
				y2 = tuple.Item2;

				_visited[tuple.Item2, tuple.Item1] = true;
				queue.Add(tuple);
			}

			v1 = new Tuple<int, int>(x2 - x1, y2 - y1);
			x_a = x1;
			y_a = y1;

			x1 = -1;
			y1 = -1;
			x2 = -1;
			y2 = -1;

			foreach (var tuple in _section.Where(tuple => tuple.Item1 != -1))
			{
				if (x1 == -1)
				{
					x1 = tuple.Item1;
					y1 = tuple.Item2;
				}

				x2 = tuple.Item1;
				y2 = tuple.Item2;

				_visited[tuple.Item2, tuple.Item1] = true;
			}

			v2 = new Tuple<int, int>(x2 - x1, y2 - y1);

			if (v1.Item1*v2.Item1 + v1.Item2*v2.Item2 < 0)
			{
				x1 = x2;
				y1 = y2;
				v1 = new Tuple<int, int>(-v1.Item1, -v1.Item2);
			}

			while (queue.Count > 0)
			{
				var cX = queue[0].Item1;
				var cY = queue[0].Item2;
				queue.RemoveAt(0);

				for (int i = -1; i < 2; i++)
					for (int j = -1; j < 2; j++)
					{
						if (i == 0 && j == 0) continue;

						var x = cX + i;
						var y = cY + j;

						if (OutOfImage(x, y) || _visited[y, x] || _image[y, x] > 15) continue;

						var pointV1 = new Tuple<int, int>(x_a - x, y_a - y);
						var pointV2 = new Tuple<int, int>(x1 - x, y1 - y);

						var skew1 = v1.Item1*pointV1.Item2 - pointV1.Item1*v1.Item2 >= 0 ? 1 : -1;
						var skew2 = v2.Item1*pointV2.Item2 - pointV2.Item1*v2.Item2 >= 0 ? 1 : -1;

						if (skew1*skew2 < 0)
						{
							queue.Add(new Tuple<int, int>(x, y));
							_visited[y, x] = true;
						}
					}
			}
		}

		private MinutiaTypes CheckStopCriteria(int threshold = 20)
		{
			var x = _section[_sectionsCenter].Item1;
			var y = _section[_sectionsCenter].Item2;

			if (_visited[y, x])
				return MinutiaTypes.Intersection;

			if (_image[y, x] > threshold)
				return MinutiaTypes.LineEnding;

			return MinutiaTypes.NotMinutia;
		}

		//for visualisation results
		private void MakeTestBmp(int x, int y)
		{
			var image = new int[_visited.GetLength(0), _visited.GetLength(1)];

			for (var i = 0; i < image.GetLength(1); i++)
				for (var j = 0; j < image.GetLength(0); j++)
					if (_visited[j, i])
						image[j, i] = 255;

			var bmp = ImageHelper.SaveArrayToBitmap(image);

			bmp.SetPixel(x, image.GetLength(0) - y - 1, Color.Red);

			bmp.Save("Test" + _countTest + ".bmp");
			_countTest++;
		}

		private void FollowLine(Tuple<int, int> point, Directions direction)
		{
			_direction = direction;

			NewSection(point);
			if (_section[_sectionsCenter].Item1 == -1) return;

			MinutiaTypes minutiaType;
			int x, y;

			do
			{
				var oldSection = new Tuple<int, int>[_lengthWings*2 + 1];
				for (var i = 0; i < _section.Length; i++)
					oldSection[i] = _section[i];

				point = MakeStep();

				if (point.Item1 == -1) return;

				NewSection(point);
				if (_section[_sectionsCenter].Item1 == -1) return; //?

				minutiaType = CheckStopCriteria();
				//if (minutiaType == MinutiaTypes.NotMinutia)
				Paint(oldSection);
			} while (minutiaType == MinutiaTypes.NotMinutia);

			x = point.Item1;
			y = point.Item2;

			var possMinutia = new Minutia();
			possMinutia.X = x;
			possMinutia.Y = y;
			possMinutia.Angle = Convert.ToSingle(_sectionsAngle);

			if (IsDuplicate(possMinutia, minutiaType)) return;

			if (!CheckAndDeleteFalseMinutia(possMinutia, minutiaType))
			{
				AddMinutia(possMinutia, minutiaType);
				MakeTestBmp(x, y);
			} 
		}

		private bool CheckAndDeleteFalseMinutia(Minutia minutia, MinutiaTypes minutiaTypes, double delta = 6)
		{
			var i =
				_minutias.FindIndex(x => Math.Sqrt(Math.Pow(x.Item1.X - minutia.X, 2) + Math.Pow(x.Item1.Y - minutia.Y, 2)) < delta);

			if (i == -1) return false;

			_minutias.RemoveAt(i);
			return true;
		}

		private bool IsDuplicate(Minutia minutia, MinutiaTypes minutiaTypes, double delta = 6)
		{
			return
				_minutias.Exists(
					x =>
						(Math.Sqrt(Math.Pow(x.Item1.X - minutia.X, 2) + Math.Pow(x.Item1.Y - minutia.Y, 2)) < delta) &&
						(x.Item2 == minutiaTypes));
		}

		public bool[,] GetVisitedMap()
		{
			return _visited;
		}

		public List<Tuple<Minutia, MinutiaTypes>> GetMinutiaList()
		{
			return _minutias;
		}
	}
}