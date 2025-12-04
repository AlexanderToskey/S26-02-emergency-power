
import customtkinter

from PIL import Image

class MyCheckboxFrame(customtkinter.CTkFrame):
    def __init__(self, master):
        super().__init__(master)

        self.checkbox_1 = customtkinter.CTkCheckBox(self, text="Include Outage Prediction")
        self.checkbox_1.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="w")
        self.checkbox_2 = customtkinter.CTkCheckBox(self, text="Include Outage Duration")
        self.checkbox_2.grid(row=1, column=0, padx=10, pady=(10, 0), sticky="w")
        self.checkbox_3 = customtkinter.CTkCheckBox(self, text="Include Outage Scope")
        self.checkbox_3.grid(row=2, column=0, padx=10, pady=(10, 0), sticky="w")
        self.checkbox_4 = customtkinter.CTkCheckBox(self, text="Include Explanation")
        self.checkbox_4.grid(row=3, column=0, padx=10, pady=(10, 10), sticky="w")
        
    def get(self):
        checked_checkboxes = []
        if self.checkbox_1.get() == 1:
            checked_checkboxes.append(self.checkbox_1.cget("text"))
        if self.checkbox_2.get() == 1:
            checked_checkboxes.append(self.checkbox_2.cget("text"))
        if self.checkbox_3.get() == 1:
            checked_checkboxes.append(self.checkbox_3.cget("text"))
        if self.checkbox_4.get():
            checked_checkboxes.append(self.checkbox_4.cget("text"))
        return checked_checkboxes

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.title("Emergency Power Response Reporting Dashboard")
        self.geometry("1000x700")

        self.button_Outage_Report = customtkinter.CTkButton(self, text="Generate Outage Report", command=self.button_callbck_outage, width=100)
        self.button_Check_Connection = customtkinter.CTkButton(self, text="Check Cloud Connection", command=self.button_callbck_cloud, width=100)

        self.button_Outage_Report.grid(row=2, column=0, padx=(20,20), pady=(20, 20), sticky="ew")
        self.button_Check_Connection.grid(row=3, column=0, padx=(20,20), pady=20, sticky="ew")

        self.checkbox_frame = MyCheckboxFrame(self)
        self.checkbox_frame.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="nsw")

        self.optionmenu = customtkinter.CTkOptionMenu(self, values=["Select Region", "NRV", "Shenandoah Valley", "Richmond", "Norfolk", "NOVA"], command=self.optionmenu_callback)
        self.optionmenu.set("Select Region")
        self.optionmenu.grid(row=1, column=0, padx=20, pady=10, sticky="w")

        self.map_image = customtkinter.CTkImage(light_image=Image.open("US_Map.png"), size=(650, 400))
        self.image_label = customtkinter.CTkLabel(self, image=self.map_image, text="US Map")  # display image with a CTkLabel
        self.image_label.grid(row=0, column=1, pady=20, sticky="w")

    def button_callbck_outage(self):
        print("Outage report generated in region: " + self.optionmenu.get())
        print(self.checkbox_frame.get())

    def button_callbck_cloud(self):
        print("Connection is stable")

    def optionmenu_callback(_, choice):
        print("optionmenu dropdown clicked:", choice)
        


app = App()
app.grid_columnconfigure(0, weight=1)

app.mainloop()



