import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

class EmailSpamDetectionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Email Spam Detection")
        self.geometry("400x450")
        
        self.file_path = ""
        self.model = None
        
        self.vectorizer = CountVectorizer(stop_words='english') 
        
        # Create button to upload file
        self.upload_button = tk.Button(self, text="Upload CSV", command=self.upload_file)
        self.upload_button.pack(pady=20)
        
        # Create label to display results
        self.result_label = tk.Label(self, text="")
        self.result_label.pack(pady=10)
        
        # Additional functionality - Display Metrics
        self.display_metrics_button = tk.Button(self, text="Display Metrics", command=self.display_metrics)
        self.display_metrics_button.pack()
        
        # Button to save the trained model
        self.save_model_button = tk.Button(self, text="Save Model", command=self.save_model)
        self.save_model_button.pack(pady=10)
    
    # Function to upload file
    def upload_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=(("CSV files", "*.csv"),))
        if self.file_path:
            self.train_model()
    
    # Function to train the model
    def train_model(self):
        try:
            emails = pd.read_csv(self.file_path)
            X_train, X_test, y_train, y_test = train_test_split(emails['text'], emails['spam'], random_state=42)
            vectorizer = CountVectorizer(stop_words='english')
            X_train_vectorized = vectorizer.fit_transform(X_train)
            
            self.model = MultinomialNB()
            self.model.fit(X_train_vectorized, y_train)
            
            self.result_label.config(text="Model trained successfully!")
        except Exception as e:
            self.result_label.config(text="Error training the model.")
    
    # Function to display additional metrics
    def display_metrics(self):
        if self.model is None:
            messagebox.showerror("Error", "Please upload and train the model first.")
            return
        self.vectorizer = CountVectorizer(stop_words='english') 
        if self.file_path:
            try:
                emails = pd.read_csv(self.file_path)
                X_train, X_test, y_train, y_test = train_test_split(emails['text'], emails['spam'], random_state=42)
                X_test_vectorized = vectorizer.transform(X_test)
                y_pred = self.model.predict(X_test_vectorized)
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                
                metrics_info = f"Accuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1 Score: {f1:.2f}"
                
                self.result_label.config(text=metrics_info)
            except Exception as e:
                self.result_label.config(text="Error processing the file.")
    
    # Function to save the trained model
    def save_model(self):
        if self.model is None:
            messagebox.showerror("Error", "No model available to save.")
            return
        
        file_path = filedialog.asksaveasfilename(defaultextension=".pkl", filetypes=(("Pickle files", "*.pkl"),))
        if file_path:
            try:
                joblib.dump(self.model, file_path)
                messagebox.showinfo("Success", "Model saved successfully.")
            except Exception as e:
                messagebox.showerror("Error", "Failed to save the model.")
    
if __name__ == "__main__":
    app = EmailSpamDetectionApp()
    app.mainloop()
