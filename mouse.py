def select_pixel(event):
    print(self.true_image.getpixel((event.x, event.y)))


def set_pixel(event):
    self.x_text.delete(0, len(self.x_text.get()))
    self.y_text.delete(0, len(self.y_text.get()))
    self.x_text.insert(0, event.x)
    self.y_text.insert(0, event.y)


def set_area(event):
    print(str(event.x) + " " + str(event.y))
    if self.release:
        self.x_start = event.x
    else:
        self.x_finish = event.x

    if self.release:
        self.y_start = event.y
        self.release = False;
    else:
        self.y_finish = event.y

    self.canvas.coords(self.canvas.rect, self.x_start, self.y_start, self.x_finish, self.y_finish);


def release_left(event,self):
    self.release = True;
    self.c
