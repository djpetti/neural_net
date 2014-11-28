#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "logger.h"

namespace helpers {

const char *Logger::string_levels_ [5] = {
  "DEBUG",
  "INFO",
  "WARNING",
  "ERROR",
  "FATAL"
};
Logger *Logger::root_logger_ = nullptr;
Level Logger::printlevel_ = Level::ERROR;

Logger::Logger(const char *filename) :
    file_(fopen(filename, "w")) {}

Logger::~Logger() {
  fclose(file_);
}

Logger *Logger::GetRoot() {
  if (root_logger_ == nullptr) {
    root_logger_ = new Logger(kLogFileBaseName);
  }
  return root_logger_;
}

int Logger::Write(const char *file, int line,
    Level level, const char *format, ...) {
  if (!file_) {
    return -1;
  }

  va_list args;
  va_start(args, format);

  // Get the message in string form.
  char *message;
  vasprintf(&message, format, args);

  // Format a complete line in the log.
  time_t rawtime;
  time(&rawtime);
  char *time = ctime(&rawtime);
  // Get rid of the newline.
  int i = 0;
  while (true) {
    if (time[i] == '\n') {
      time[i] = '\0';
      break;
    }
    ++i;
  }
  const char *string_level = string_levels_[static_cast<int>(level)];
  char *to_write;
  asprintf(&to_write, "[%s %s %s:%d] %s\n",
      string_level, time, file, line, message);

  int status = fprintf(file_, "%s", to_write);
  if (status >= 0) {
    bytes_written_ += status;
  }
  fflush(file_);
  if (static_cast<int>(level) >= static_cast<int>(printlevel_)) {
    printf("%s", to_write);
  }

  free(message);
  free(to_write);
  if (level == Level::FATAL) {
    // Fatal error, abort!
    exit(1);
  }

  char *new_name;
  char date[11 + 1 + 8];
  tm *timeinfo;
  timeinfo = localtime(&rawtime);
  strftime(date, 11 + 1 + 8, "%F_%T", timeinfo);
  asprintf(&new_name, "%s.%s", kLogFileBaseName, date);
  if (bytes_written_ >= kMaxFileBytes) {
    // Start a new log file.
    fclose(file_);
    rename(kLogFileBaseName, new_name);
    file_ = fopen(kLogFileBaseName, "w");
    bytes_written_ = 0;
  }
  free(new_name);

  return status;
}

const char *Logger::kLogFileBaseName = "neural_net.log";

} // helpers
