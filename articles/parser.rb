require 'open-uri'
require 'nokogiri'
require 'byebug'
require 'action_view'

include ActionView::Helpers::SanitizeHelper

def save(good_or_bad, text)
  return unless text

  prepared_text = text.gsub(/\s\s+/, '').gsub(/\n/, ' ')
  prepared_text = "#{strip_tags prepared_text}\n"

  File.open("#{good_or_bad}.articles", 'a') { |file| file.write prepared_text }
end

File.readlines('bad.links.prepared').each do |url|
  url = url.strip
  puts "Parsing #{url}..."
  doc = open(URI(url)).read

  result = ''
  nodes = Nokogiri::HTML(doc).css('#article-content>.content>*')

  nodes.each do |child|
    next if %w[script br img].include? child.name

    what_to_append = if child['class']&.include?('date')
                       '. '
                     else
                       child.text
                     end

    result << what_to_append
  end

  save 'bad', result
end
