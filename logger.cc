#include <stdarg.h>
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

Logger::Logger(const char *filename) :
    file_(fopen(filename, "w")),
    printlevel_(Level::ERROR) {}

Logger::~Logger() {
  fclose(file_);
}

int Logger::Write(const char *file, const char *line,
      Level level, const char *format, ...) {
      va_list args;
      va_start(args, format);

      // Get the message in string form.
      char *message;
      vasprintf(&message, format, args);

      // Format a complete line in the log.
      time_t rawtime;
      time(&rawtime);
      char *time = ctime(&rawtime);
      const char *string_level = string_levels_[static_cast<int>(level)];
      char *to_write;
      asprintf(&to_write, "[%s %s %s:%s] %s\n",
          string_level, time, file, line, message);

      int status = fprintf(file_, "%s", to_write);
      if (static_cast<int>(level) >= static_cast<int>(printlevel_)) {
        printf("%s", to_write);
      }

      free(message);
      free(to_write);
      return status;
} 

} // helpers
