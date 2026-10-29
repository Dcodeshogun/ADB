"""Interactive MAX7219 16-column orientation calibration.

Close adb.py and the Arduino Serial Monitor before running this script.
The NodeMCU firmware must accept M:<16 bits>, with 0=ON and 1=OFF.
"""

import time
import sys

try:
    import serial
except ImportError:
    print("pyserial is missing. Install it with: python -m pip install pyserial")
    raise SystemExit(1)

PORT = "COM4"
BAUD_RATE = 115200
NUM_COLUMNS = 16


def make_mask(active_column=None):
    # All columns OFF except the one being tested.
    # Firmware protocol: 0 = ON, 1 = OFF.
    bits = ["1"] * NUM_COLUMNS
    if active_column is not None:
        bits[active_column] = "0"
    return "M:" + "".join(bits) + "\n"


def main():
    try:
        board = serial.Serial(PORT, BAUD_RATE, timeout=0.2)
    except (serial.SerialException, OSError) as exc:
        print(f"Could not open {PORT}: {exc}")
        print("Close adb.py / Arduino Serial Monitor and verify the COM port.")
        return 1

    try:
        time.sleep(2)  # ESP8266 may reset when serial opens.
        board.reset_input_buffer()

        print("\nMAX7219 orientation calibration")
        print("All columns will be blank except for one full lit column.")
        print("Start by setting FLIP_MATRIX_2 false in the Arduino firmware")
        print("if you want to discover the modules' raw physical orientation.\n")
        input("Arrange the modules as you want them. Press Enter to start...")

        for col in range(NUM_COLUMNS):
            board.write(make_mask(col).encode("ascii"))
            board.flush()
            time.sleep(0.15)
            print(f"\nLogical column {col:02d} is ON.")
            input("Write down which physical column lit, then press Enter...")

        print("\nCalibration complete. Restoring all columns ON.")
        board.write(("M:" + "0" * NUM_COLUMNS + "\n").encode("ascii"))
        board.flush()
        time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nInterrupted. Restoring all columns ON.")
        try:
            board.write(("M:" + "0" * NUM_COLUMNS + "\n").encode("ascii"))
            board.flush()
        except (serial.SerialException, OSError):
            pass
    finally:
        board.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
