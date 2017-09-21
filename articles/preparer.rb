require 'action_view'
require 'csv'
require 'byebug'

include ActionView::Helpers::SanitizeHelper

files = %w[
  content_with_rubrics_labels_20150203153500_20160315154634.csv
  content_with_rubrics_labels_20140416180356_20140717154556.csv
  content_with_rubrics_labels_20160422163729_20160517170149.csv
  content_with_rubrics_labels_20140814140207_20141229122650.csv
  content_with_rubrics_labels_20160712093420_20170328171030.csv
]

def save(good_or_bad, text)
  return unless text

  prepared_text = text
                    .gsub(/\n#.*/, '')   # remove all headings
                    .gsub(/box#\d+/, '') # remove all boxes
                    .gsub(/\n/, '')      # remove all new lines

  prepared_text = "#{strip_tags prepared_text}\n"

  File.open("#{good_or_bad}.articles", 'a') { |file| file.write prepared_text }
end

matches = {}

files.each do |file|
  CSV.foreach file do |row|
    labels = row[4].split(',')
    labels.each do |label|
      if matches[label]
        matches[label] = matches[label] + 1
      else
        matches[label] = 1
        puts label
      end

      save 'good', row[6] if label =~ /(goodnews|good_news)/
    end
  end
end

puts matches
