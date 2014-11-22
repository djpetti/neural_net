#ifndef NEURAL_NET_LOGGER_H_
#define NEURAL_NET_LOGGER_H_

#include <stdio.h>

#include "macros.h"

// Logging macros, similar to FRC team 971's logging implementation. (Which I
// think is itself based off of glog...)
// The "##" is a special gcc thing that means "delete the previous comma if we
// get no arguments after format."
#define LOG(level, format, ...) \
    helpers::Logger::GetRoot()->Write( \
    __FILE__, __LINE__, level, format, ##__VA_ARGS__);
#ifndef NDEBUG
#define CHECK(condition, format, ...) \
    do { \
        if (!(condition)) { \
          LOG(Level::FATAL, format, ##__VA_ARGS__); \
        } \
    } while (false)
#else
#define CHECK(condition, format, ...) do { } while (false)
#endif

// Enum for logging levels.
enum class Level {
  DEBUG = 0,
  INFO,
  WARNING,
  ERROR,
  FATAL
};

namespace helpers {

// A class that handles writing log messages to a file.
class Logger {
 public:
   ~Logger();
  // Returns a the root logger, creates a new root logger if none exists.
  static Logger *GetRoot();
  // Writes a log message to the file. The "file" and "line" args are
  // automatically filled in by the LOG macro.
  int Write(const char *file, int line, Level level, const char *format, ...);
  // Writes any messages <level> and above to stdout. (Messages of level ERROR
  // and above are automatically writter to stderr.)
  inline static void Show(Level level) {
    printlevel_ = level;
  }

  DISSALOW_COPY_AND_ASSIGN(Logger);

 private:
  // How many bytes we'll write before we split out a new file.
  const int kMaxFileBytes = 10000000;
  // The name we use for our log file.
  static const char *kLogFileBaseName;

  // Constructor requires the name of the logfile.
  explicit Logger(const char *filename);
  static Logger *root_logger_;
  // The file that we're writing to.
  FILE *file_;
  // An array that facilitates converting levels to strings.
  static const char *string_levels_ [5];
  // The lowest level to print to stdout.
  static Level printlevel_;
  // How many bytes we've written to this file.
  int bytes_written_ = 0;
};

} // helpers

#endif
