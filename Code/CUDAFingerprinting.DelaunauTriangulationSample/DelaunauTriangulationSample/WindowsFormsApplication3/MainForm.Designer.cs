namespace DelaunauTriangulation
{
    partial class MainForm
    {
        /// <summary>
        /// Обязательная переменная конструктора.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Освободить все используемые ресурсы.
        /// </summary>
        /// <param name="disposing">истинно, если управляемый ресурс должен быть удален; иначе ложно.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Код, автоматически созданный конструктором форм Windows

        /// <summary>
        /// Требуемый метод для поддержки конструктора — не изменяйте 
        /// содержимое этого метода с помощью редактора кода.
        /// </summary>
        private void InitializeComponent()
        {
            this.Count = new System.Windows.Forms.TrackBar();
            this.Delay = new System.Windows.Forms.TrackBar();
            this.CountLabel = new System.Windows.Forms.Label();
            this.DelayLabel = new System.Windows.Forms.Label();
            this.CountVolume = new System.Windows.Forms.Label();
            this.DelayVolume = new System.Windows.Forms.Label();
            ((System.ComponentModel.ISupportInitialize)(this.Count)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.Delay)).BeginInit();
            this.SuspendLayout();
            // 
            // Count
            // 
            this.Count.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.Count.Location = new System.Drawing.Point(457, 12);
            this.Count.Maximum = 1000;
            this.Count.Minimum = 3;
            this.Count.Name = "Count";
            this.Count.Size = new System.Drawing.Size(104, 45);
            this.Count.TabIndex = 0;
            this.Count.Value = 3;
            this.Count.Scroll += new System.EventHandler(this.Count_Scroll);
            // 
            // Delay
            // 
            this.Delay.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.Delay.Location = new System.Drawing.Point(457, 79);
            this.Delay.Maximum = 200;
            this.Delay.Minimum = 1;
            this.Delay.Name = "Delay";
            this.Delay.Size = new System.Drawing.Size(104, 45);
            this.Delay.TabIndex = 1;
            this.Delay.Value = 100;
            this.Delay.Scroll += new System.EventHandler(this.Delay_Scroll);
            // 
            // CountLabel
            // 
            this.CountLabel.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.CountLabel.AutoSize = true;
            this.CountLabel.Location = new System.Drawing.Point(385, 12);
            this.CountLabel.Name = "CountLabel";
            this.CountLabel.RightToLeft = System.Windows.Forms.RightToLeft.No;
            this.CountLabel.Size = new System.Drawing.Size(66, 13);
            this.CountLabel.TabIndex = 2;
            this.CountLabel.Text = "Количество";
            // 
            // DelayLabel
            // 
            this.DelayLabel.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.DelayLabel.AutoSize = true;
            this.DelayLabel.Location = new System.Drawing.Point(393, 79);
            this.DelayLabel.Name = "DelayLabel";
            this.DelayLabel.Size = new System.Drawing.Size(58, 13);
            this.DelayLabel.TabIndex = 3;
            this.DelayLabel.Text = "Задержка";
            // 
            // CountVolume
            // 
            this.CountVolume.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.CountVolume.AutoSize = true;
            this.CountVolume.Location = new System.Drawing.Point(493, 44);
            this.CountVolume.Name = "CountVolume";
            this.CountVolume.Size = new System.Drawing.Size(35, 13);
            this.CountVolume.TabIndex = 4;
            this.CountVolume.Text = "label3";
            // 
            // DelayVolume
            // 
            this.DelayVolume.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.DelayVolume.AutoSize = true;
            this.DelayVolume.Location = new System.Drawing.Point(493, 111);
            this.DelayVolume.Name = "DelayVolume";
            this.DelayVolume.Size = new System.Drawing.Size(35, 13);
            this.DelayVolume.TabIndex = 5;
            this.DelayVolume.Text = "label4";
            this.DelayVolume.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // MainForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.BackColor = System.Drawing.Color.White;
            this.ClientSize = new System.Drawing.Size(573, 441);
            this.Controls.Add(this.DelayVolume);
            this.Controls.Add(this.CountVolume);
            this.Controls.Add(this.DelayLabel);
            this.Controls.Add(this.CountLabel);
            this.Controls.Add(this.Delay);
            this.Controls.Add(this.Count);
            this.Name = "MainForm";
            this.Text = "Triangulation";
            this.ResizeEnd += new System.EventHandler(this.MainForm_ResizeEnd);
            this.MouseClick += new System.Windows.Forms.MouseEventHandler(this.MainForm_MouseClick);
            this.Resize += new System.EventHandler(this.MainForm_Resize);
            ((System.ComponentModel.ISupportInitialize)(this.Count)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.Delay)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.TrackBar Count;
        private System.Windows.Forms.TrackBar Delay;
        private System.Windows.Forms.Label CountLabel;
        private System.Windows.Forms.Label DelayLabel;
        private System.Windows.Forms.Label CountVolume;
        private System.Windows.Forms.Label DelayVolume;

    }
}

