import pygame
import numpy as np
import pickle

# Load the trained model
with open("xgb_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Initialize Pygame
pygame.init()

# Window settings
WIDTH, HEIGHT = 600, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Home Price Predictor")

# Colors
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
BLACK = (0, 0, 0)
BLUE = (100, 149, 237)

# Font
font = pygame.font.Font(None, 28)

# Define input fields
fields = [
    "LivingArea", "YearBuilt", "BathroomsTotalInteger", "LotSizeSquareFeet",
    "DaysOnMarket", "MLSAreaMajor", "HighSchoolDistrict", "GarageSpaces",
    "Stories", "SubdivisionName", "FireplacesTotal", "AssociationFee", "TaxAnnualAmount"
]
inputs = {field: "" for field in fields}
binary_fields = ["PoolPrivateYN", "ViewYN", "AttachedGarageYN"]
binary_inputs = {field: 0 for field in binary_fields}  # Default to 0 (No)

# Input field positions
input_rects = []
y_offset = 40
for i, field in enumerate(fields):
    input_rects.append(pygame.Rect(200, y_offset + (i * 30), 150, 25))

binary_rects = []
for i, field in enumerate(binary_fields):
    binary_rects.append(pygame.Rect(200, y_offset + (len(fields) * 30) + (i * 30), 25, 25))

# Button for prediction
predict_button = pygame.Rect(200, y_offset + (len(fields) * 30) + (len(binary_fields) * 30) + 20, 150, 40)
predicted_price = ""

# Main loop
running = True
active_field = None
while running:
    screen.fill(WHITE)

    # Draw input labels and boxes
    for i, field in enumerate(fields):
        text = font.render(field, True, BLACK)
        screen.blit(text, (10, y_offset + (i * 30)))
        pygame.draw.rect(screen, GRAY if active_field == field else BLACK, input_rects[i], 2)

        # Display user input
        input_surface = font.render(inputs[field], True, BLACK)
        screen.blit(input_surface, (input_rects[i].x + 5, input_rects[i].y + 5))

    # Draw binary checkboxes
    for i, field in enumerate(binary_fields):
        text = font.render(field, True, BLACK)
        screen.blit(text, (10, y_offset + (len(fields) * 30) + (i * 30)))
        pygame.draw.rect(screen, BLACK, binary_rects[i], 2)
        if binary_inputs[field]:  # If checked
            pygame.draw.line(screen, BLACK, (binary_rects[i].x, binary_rects[i].y),
                             (binary_rects[i].x + 25, binary_rects[i].y + 25), 3)
            pygame.draw.line(screen, BLACK, (binary_rects[i].x + 25, binary_rects[i].y),
                             (binary_rects[i].x, binary_rects[i].y + 25), 3)

    # Draw predict button
    pygame.draw.rect(screen, BLUE, predict_button)
    text = font.render("Predict", True, WHITE)
    screen.blit(text, (predict_button.x + 40, predict_button.y + 10))

    # Display predicted price
    if predicted_price:
        price_text = font.render(f"Predicted Price: ${predicted_price}", True, BLACK)
        screen.blit(price_text, (150, predict_button.y + 50))

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Handle clicking input fields
        if event.type == pygame.MOUSEBUTTONDOWN:
            active_field = None
            for i, rect in enumerate(input_rects):
                if rect.collidepoint(event.pos):
                    active_field = fields[i]
                    break
            for i, rect in enumerate(binary_rects):
                if rect.collidepoint(event.pos):
                    binary_inputs[binary_fields[i]] = 1 - binary_inputs[binary_fields[i]]  # Toggle checkbox

            # Handle predict button click
            if predict_button.collidepoint(event.pos):
                try:
                    input_values = [float(inputs[field]) if inputs[field] else 0 for field in fields]
                    binary_values = [binary_inputs[field] for field in binary_fields]
                    model_input = np.array(input_values + binary_values).reshape(1, -1)

                    # Make prediction
                    predicted_price = np.expm1(model.predict(model_input)[0])  # Convert back if trained in log-space
                    predicted_price = round(predicted_price, 2)
                except Exception as e:
                    predicted_price = "Error"

        # Handle text input
        if event.type == pygame.KEYDOWN and active_field:
            if event.key == pygame.K_BACKSPACE:
                inputs[active_field] = inputs[active_field][:-1]  # Remove last character
            elif event.unicode.isdigit() or event.unicode in ".-":  # Allow numbers and decimal
                inputs[active_field] += event.unicode

    pygame.display.flip()

pygame.quit()
