namespace OptimumConversionOfTriangles
{
    partial class MainForm
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.CanRenamePoints = new System.Windows.Forms.CheckBox();
            this.SuspendLayout();
            // 
            // CanRenamePoints
            // 
            this.CanRenamePoints.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.CanRenamePoints.AutoSize = true;
            this.CanRenamePoints.Location = new System.Drawing.Point(66, 12);
            this.CanRenamePoints.Name = "CanRenamePoints";
            this.CanRenamePoints.Size = new System.Drawing.Size(206, 17);
            this.CanRenamePoints.TabIndex = 0;
            this.CanRenamePoints.Text = "Имеет ли значение порядок точек?";
            this.CanRenamePoints.UseVisualStyleBackColor = true;
            // 
            // MainForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.BackColor = System.Drawing.Color.White;
            this.ClientSize = new System.Drawing.Size(284, 261);
            this.Controls.Add(this.CanRenamePoints);
            this.Name = "MainForm";
            this.Text = "ConversionOfTriangles";
            this.ResizeEnd += new System.EventHandler(this.OnChangeSize);
            this.MouseClick += new System.Windows.Forms.MouseEventHandler(this.MainForm_MouseClick);
            this.Resize += new System.EventHandler(this.OnChangeSize);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.CheckBox CanRenamePoints;
    }
}

